"""PK-DB API loader — pull *real, cited* pharmacokinetic data from pk-db.com.

PK-DB (https://pk-db.com, Grzegorzewski et al., Nucleic Acids Res 2020,
doi:10.1093/nar/gkaa990) is an open pharmacokinetics database with a REST API.
This module is the reproducible bridge from that API to PharmacometricsBench:
it downloads a drug's curated record and normalises it into provenance-carrying
records that a task generator can turn into graded tasks.

Access reality (verified against the live API, 2026-07-12)
----------------------------------------------------------
The API is a **two-step** service:

1. ``GET /api/v1/filter/?<dimension>__<field>=<value>&download=true`` builds a
   filtered set and streams back a ZIP of CSVs (studies / interventions /
   individuals / groups / outputs / ...).
2. The working entry filters are ``studies__substance`` and
   ``interventions__substance`` (by substance *name*).

What the **anonymous** download exposes, verified per drug:

* ``studies.csv``      — study metadata + reference (name, sid, licence, access).
* ``interventions.csv``— **dosing**: substance, dose value+unit, route, form.
  Real and cited. Abundant (hundreds of oral regimens across the seed drugs).
* ``individuals.csv`` / ``groups.csv`` — subject *characteristica* (age, weight,
  sex, genotype) plus a few covariate measurements. The ``clearance`` rows here
  are dominated by **creatinine clearance** (``substance == 'creatinine'``), i.e.
  a renal covariate, NOT the drug's clearance.
* ``outputs.csv`` — **empty** for anonymous requests, and every ``outputs__*``
  filter zeroes the result set. So the drug-level PK parameters (drug clearance,
  AUC, Cmax, t1/2) — which PK-DB *does* hold and openly licenses — are not
  reachable via the anonymous REST path. They require an authenticated account
  token (free PK-DB account), the web UI, or a Zenodo snapshot.

Consequences for the loader
---------------------------
* Anonymously it harvests a real, cited **dosing + study catalogue**.
* Given ``PKDB_API_TOKEN`` (or an explicit token), it additionally attempts the
  **PK-parameter** harvest (``harvest_pk_parameters``) so the answer-key path is
  built and ready — it activates the moment a token is supplied.
* Every harvested value carries provenance (study sid/name, reference, licence).
  Missing fields are skipped with a logged reason; nothing is fabricated.

CLI::

    python -m pharmacometricsbench.pkdb.loader --drugs caffeine paracetamol
    python -m pharmacometricsbench.pkdb.loader --coverage       # seed-set sweep
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import os
import ssl
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

API_URL = "https://pk-db.com/api/v1"

# Tissues that count as systemic drug exposure (exclude urine/faeces/saliva).
_PLASMA_TISSUES = {"plasma", "serum", "blood"}

# Real drug-disposition measurement types (what a genuine answer key would use).
# Covariate look-alikes ('gfr', renal 'clearance' of creatinine) are excluded by
# also requiring ``substance == <drug sid>`` in :func:`harvest_pk_parameters`.
PK_PARAMETER_TYPES = {
    "auc_inf", "auc_end", "cmax", "tmax", "thalf",
    "clearance", "clearance/bioavailability",
    "vd", "vd/bioavailability", "mrt", "kel",
}

# Well-known INN/USAN name divergences → PK-DB's preferred label name (lowercase).
# Used only for exact-name resolution; extend as the seed set grows.
_SYNONYMS: dict[str, str] = {
    "acetaminophen": "paracetamol",
}

# Seed set proposed in FIH_PBPK_dataset_sourcing_plan.md (PK-DB ∩ common PBPK cpds).
SEED_DRUGS = [
    "caffeine", "midazolam", "paracetamol", "diazepam", "theophylline",
    "omeprazole", "simvastatin", "digoxin", "torasemide", "codeine",
    "verapamil", "sildenafil", "propranolol", "fluconazole", "felodipine",
]


# ── Data model ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Reference:
    """Where a value came from — always attached, never optional in output."""
    study_sid: str
    study_name: str
    licence: str
    access: str

    def to_dict(self) -> dict[str, Any]:
        return {"study_sid": self.study_sid, "study_name": self.study_name,
                "licence": self.licence, "access": self.access}


@dataclass(frozen=True)
class Dosing:
    substance: str
    dose: float
    unit: str
    route: str
    form: str
    reference: Reference

    def to_dict(self) -> dict[str, Any]:
        return {"substance": self.substance, "dose": self.dose, "unit": self.unit,
                "route": self.route, "form": self.form,
                "reference": self.reference.to_dict()}


@dataclass(frozen=True)
class PKParameter:
    """A real drug-level PK measurement (answer-key material)."""
    substance: str
    measurement_type: str
    value: float
    unit: str
    tissue: str
    n: int | None
    reference: Reference
    route: str = ""     # "" when the export does not carry route for this row

    def to_dict(self) -> dict[str, Any]:
        return {"substance": self.substance, "measurement_type": self.measurement_type,
                "value": self.value, "unit": self.unit, "tissue": self.tissue,
                "n": self.n, "route": self.route, "reference": self.reference.to_dict()}


@dataclass
class DrugHarvest:
    drug: str
    studies: list[Reference] = field(default_factory=list)
    dosing: list[Dosing] = field(default_factory=list)
    pk_parameters: list[PKParameter] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)     # human-readable skip reasons

    def coverage(self) -> dict[str, Any]:
        from collections import Counter
        return {
            "drug": self.drug,
            "n_studies": len(self.studies),
            "n_open_studies": sum(1 for r in self.studies if r.licence == "open"),
            "n_dosing": len(self.dosing),
            "oral_dosing": sum(1 for d in self.dosing if d.route == "oral"),
            "n_pk_parameters": len(self.pk_parameters),
            "pk_by_type": dict(Counter(p.measurement_type for p in self.pk_parameters)),
            "n_skipped": len(self.skipped),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug": self.drug,
            "studies": [r.to_dict() for r in self.studies],
            "dosing": [d.to_dict() for d in self.dosing],
            "pk_parameters": [p.to_dict() for p in self.pk_parameters],
            "skipped": self.skipped,
            "coverage": self.coverage(),
        }


# ── Client ──────────────────────────────────────────────────────────────────
class PKDBClient:
    """Thin, dependency-free client over the PK-DB REST API.

    ``token`` (or ``$PKDB_API_TOKEN``) is sent as an ``Authorization`` header and
    unlocks the PK-parameter (answer-key) harvest. ``open_only`` drops studies
    whose licence is not ``open``. ``verify_tls=False`` mirrors the corporate-proxy
    escape hatch used elsewhere in this repo; leave it True in production.
    """

    def __init__(self, token: str | None = None, *, open_only: bool = True,
                 timeout: int = 120, verify_tls: bool = True) -> None:
        self.token = token or os.environ.get("PKDB_API_TOKEN")
        self.open_only = open_only
        self.timeout = timeout
        self._ctx: ssl.SSLContext | None = None
        if not verify_tls:
            self._ctx = ssl.create_default_context()
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            # PK-DB uses DRF TokenAuthentication.
            h["Authorization"] = f"Token {self.token}"
        return h

    def fetch_filter_zip(self, dimension: str, value: str) -> zipfile.ZipFile:
        """Two-step download: ``/filter/?<dimension>__substance=<value>&download=true``.

        ``dimension`` is ``studies`` or ``interventions`` (the entry filters that
        work anonymously). Returns an in-memory ZipFile of the CSV tables.
        """
        q = urllib.parse.urlencode({f"{dimension}__substance": value, "download": "true"})
        url = f"{API_URL}/filter/?{q}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
            payload = r.read()
        try:
            return zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"PK-DB did not return a ZIP for {dimension}__substance={value} "
                f"(first bytes: {payload[:80]!r})"
            ) from exc

    # ── Harvest ─────────────────────────────────────────────────────────────
    def harvest_drug(self, drug: str) -> DrugHarvest:
        """Full harvest for one drug: studies + dosing (+ PK params if authed)."""
        try:
            zf = self.fetch_filter_zip("studies", drug)
        except Exception as exc:  # noqa: BLE001 — surface as a skip, not a crash
            h = DrugHarvest(drug=drug)
            h.skipped.append(f"fetch failed: {exc}")
            return h
        return self.harvest_from_zip(drug, zf)

    def harvest_from_zip(self, drug: str, zf: zipfile.ZipFile) -> DrugHarvest:
        """Parse an already-downloaded ZIP into a :class:`DrugHarvest`.

        Split out from :meth:`harvest_drug` so the parsing is unit-testable
        offline against a saved fixture without hitting the network.
        """
        h = DrugHarvest(drug=drug)
        studies = _read_csv(zf, "studies.csv")
        interventions = _read_csv(zf, "interventions.csv")
        # outputs.csv is the dedicated PK-parameter table (empty anonymously, the
        # primary answer-key source once a token is supplied); individuals/groups
        # carry per-subject/-arm measurements. Parse all three.
        rows = (_read_csv(zf, "outputs.csv")
                + _read_csv(zf, "individuals.csv")
                + _read_csv(zf, "groups.csv"))

        # Licence index from studies.csv (authoritative where present). The
        # anonymous export often ships a *partial* studies.csv, so a study
        # referenced by an intervention may be absent here — that yields
        # licence "unknown", NOT a silent drop.
        licences: dict[str, dict[str, str]] = {}
        for s in studies:
            sid = s.get("sid", "")
            if sid:
                licences[sid] = {
                    "name": s.get("name", ""),
                    "licence": (s.get("licence") or "").strip().lower(),
                    "access": (s.get("access") or "").strip().lower(),
                }

        def make_ref(sid: str, name: str) -> Reference:
            meta = licences.get(sid, {})
            return Reference(
                study_sid=sid, study_name=meta.get("name") or name,
                licence=meta.get("licence") or "unknown",
                access=meta.get("access") or "unknown",
            )

        def gate(ref: Reference, what: str) -> bool:
            """True if kept. Drop only studies KNOWN to be non-open, and log it —
            never drop silently. 'unknown' is kept (PK-DB imposes no restriction
            beyond the data owner's; see TERMS_OF_USE)."""
            if self.open_only and ref.licence not in ("open", "unknown"):
                h.skipped.append(f"{what} {ref.study_sid}: licence={ref.licence!r} (not open)")
                return False
            return True

        # Study catalogue (from studies.csv).
        for sid, meta in licences.items():
            ref = make_ref(sid, meta["name"])
            if gate(ref, "study"):
                h.studies.append(ref)

        # Resolve the drug's substance sid(s) from its own dosing rows, so PK
        # params can be filtered to the drug (excluding creatinine/co-medication).
        # FAIL CLOSED: if the sid cannot be resolved we refuse to emit anything —
        # never fall through to an unfiltered "keep everything" that would let a
        # creatinine-clearance covariate masquerade as the drug's answer key.
        drug_sids = self._drug_substance_sids(drug, interventions)
        if not drug_sids:
            h.skipped.append(
                f"could not resolve a substance sid for {drug!r} in this response "
                "— refusing to emit unfiltered dosing/PK data (fail-closed)")
            return h

        # Dosing (real, cited) — restricted to the target drug's substance sids.
        # Provenance comes from the intervention row itself, so a truncated
        # studies.csv never silently discards a real, cited dose.
        for iv in interventions:
            sub = iv.get("substance", "")
            if sub not in drug_sids:
                continue
            ref = make_ref(iv.get("study_sid", ""), iv.get("study_name", ""))
            if not gate(ref, "dosing"):
                continue
            dose = _to_float(iv.get("value") or iv.get("mean") or iv.get("median"))
            if dose is None:
                h.skipped.append(f"dosing {ref.study_sid}/{sub}: no numeric dose")
                continue
            h.dosing.append(Dosing(
                substance=sub, dose=dose, unit=iv.get("unit", ""),
                route=(iv.get("route") or "").strip().lower(),
                form=iv.get("form", ""), reference=ref,
            ))

        # PK parameters (answer keys) — present only with authentication.
        h.pk_parameters = self.harvest_pk_parameters(rows, drug_sids, make_ref, gate, h)
        if not h.pk_parameters:
            note = ("no drug-level PK parameters in this response — expected "
                    "anonymously (outputs table is auth-gated)")
            if not self.authenticated:
                note += "; supply PKDB_API_TOKEN to harvest answer keys"
            h.skipped.append(note)
        return h

    def harvest_pk_parameters(
        self, rows: list[dict[str, str]], drug_sids: set[str],
        make_ref: Any, gate: Any, h: DrugHarvest,
    ) -> list[PKParameter]:
        """Extract genuine drug-level PK parameters from individual/group rows.

        Requires ``substance == <drug sid>`` (drops creatinine-clearance and other
        covariate look-alikes), a whitelisted disposition ``measurement_type``, and
        a systemic tissue. Returns [] when the outputs are absent (anonymous case).
        ``make_ref``/``gate`` are the provenance + licence helpers built in
        :meth:`harvest_from_zip`.
        """
        out: list[PKParameter] = []
        for r in rows:
            mt = r.get("measurement_type", "")
            if mt not in PK_PARAMETER_TYPES:
                continue
            sub = r.get("substance", "")
            if sub not in drug_sids:
                continue  # e.g. creatinine clearance — a covariate, not the drug
            tissue = (r.get("tissue") or "").strip().lower()
            if tissue and tissue not in _PLASMA_TISSUES:
                continue
            if not r.get("study_sid"):
                # No study id → no valid provenance and it would collapse distinct
                # records into one pool bucket. Refuse it rather than emit a blank
                # source (honesty contract: every value is traceable).
                h.skipped.append(f"pk_param {sub}/{mt}: missing study_sid (no provenance)")
                continue
            ref = make_ref(r.get("study_sid", ""), r.get("study_name", ""))
            if not gate(ref, "pk_param"):
                continue
            value = _to_float(r.get("value") or r.get("mean") or r.get("median"))
            if value is None:
                h.skipped.append(f"pk_param {ref.study_sid}/{sub}/{mt}: no numeric value")
                continue
            out.append(PKParameter(
                substance=sub, measurement_type=mt, value=value,
                unit=r.get("unit", ""), tissue=tissue or "plasma",
                n=_to_int(r.get("count")), reference=ref,
                route=(r.get("route") or "").strip().lower(),
            ))
        return out

    @staticmethod
    def _label_head(label: str) -> str:
        """The first word of a PK-DB label ``'name (code)'`` → the substance's
        base name. e.g. ``'caffeine (137X)'`` → ``'caffeine'``,
        ``'codeine phosphate (...)'`` → ``'codeine'`` (lowercased).

        Stripping the trailing ``' (code)'`` and taking the first word matches a
        drug and its salt/ester forms without capturing a longer congener.
        """
        s = (label or "").strip().lower()
        cut = s.rfind(" (")
        s = (s[:cut] if cut != -1 else s).strip()
        return s.split()[0] if s else ""

    @classmethod
    def _drug_substance_sids(cls, drug: str, interventions: list[dict[str, str]]) -> set[str]:
        """The substance sid(s) whose label's *first word* equals the drug name.

        PK-DB dosing rows carry ``substance`` (sid, e.g. ``caf``) and
        ``substance_label`` (e.g. ``caffeine (137X)``). Matching the first word —
        rather than a substring — keeps salt/ester forms (``codeine phosphate``)
        while rejecting congeners (``dihydrocodeine``), superstrings
        (``esomeprazole`` vs ``omeprazole``), and metabolites. A small synonym map
        covers well-known INN/USAN divergences (acetaminophen → paracetamol).
        """
        drug_l = drug.strip().lower()
        wanted = {drug_l, _SYNONYMS.get(drug_l, drug_l)}
        sids: set[str] = set()
        for iv in interventions:
            sub = iv.get("substance", "")
            if sub and cls._label_head(iv.get("substance_label", "")) in wanted:
                sids.add(sub)
        return sids


# ── CSV / coercion helpers ──────────────────────────────────────────────────
def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        raw = zf.read(name).decode("utf-8", "replace")
    except KeyError:
        return []
    return list(csv.DictReader(io.StringIO(raw)))


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None  # drop NaN / ±inf


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


# ── CLI ─────────────────────────────────────────────────────────────────────
def _print_coverage(harvests: list[DrugHarvest], authed: bool) -> None:
    print(f"\nPK-DB harvest  (authenticated={authed})\n" + "-" * 60)
    tot_dose = tot_pk = 0
    for h in harvests:
        c = h.coverage()
        tot_dose += c["n_dosing"]
        tot_pk += c["n_pk_parameters"]
        pk = c["pk_by_type"] or "—"
        print(f"  {h.drug:13s} studies={c['n_open_studies']:>3}  "
              f"dosing={c['n_dosing']:>4} (oral {c['oral_dosing']:>4})  "
              f"pk_params={c['n_pk_parameters']:>3}  {pk}")
    print("-" * 60)
    print(f"  totals: dosing={tot_dose}  pk_parameters={tot_pk}")
    if tot_pk == 0:
        print("\n  NOTE: 0 drug-level PK parameters — anonymous access only exposes\n"
              "  dosing/study data. Set PKDB_API_TOKEN (free PK-DB account) to\n"
              "  harvest AUC/Cmax/clearance answer keys.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="PK-DB loader for PharmacometricsBench")
    ap.add_argument("--drugs", nargs="*", help="drug names (default: seed set)")
    ap.add_argument("--coverage", action="store_true", help="print a coverage table")
    ap.add_argument("--out", help="write harvested records as JSON to this path")
    ap.add_argument("--all-licences", action="store_true",
                    help="do not restrict to open-licence studies")
    ap.add_argument("--no-verify-tls", action="store_true")
    args = ap.parse_args()

    drugs = args.drugs or SEED_DRUGS
    client = PKDBClient(open_only=not args.all_licences, verify_tls=not args.no_verify_tls)
    harvests = [client.harvest_drug(d) for d in drugs]

    if args.coverage or not args.out:
        _print_coverage(harvests, client.authenticated)
    if args.out:
        import json
        with open(args.out, "w") as fh:
            json.dump([h.to_dict() for h in harvests], fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
