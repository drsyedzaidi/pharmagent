"""nlmixr2 adapter — a real external R engine (FeRx-style shell-out).

Fits a candidate with nlmixr2 in R, parses the population estimates back, then
scores them through the SAME ``score_from_population`` the native engine uses —
so an external fit is judged by our simulator on our data, identical footing.
Native OFV from nlmixr2 is on its own scale and is kept out of ranking.

``available()`` returns False unless ``Rscript`` and the ``nlmixr2`` package are
both present, so the runner records an ``absent`` row and skips it on any machine
without R — no crash, no dependency. v0 supports the ``oral_1cmt`` and
``iv_1cmt`` structural models; other keys return a ``failed`` result.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .base import CandidateSpec, EngineResult
from .dataset_io import write_nonmem_csv
from .scoring import score_from_population

# model_key -> R model config
_SUPPORTED: dict[str, dict[str, Any]] = {
    "oral_1cmt": {
        "is_iv": False, "struct": ["CL", "V", "KA"], "iiv": ["CL", "V"],
        "ode": ["d/dt(depot) <- -KA*depot",
                "d/dt(center) <- KA*depot - (CL/V)*center"],
        "cp": "cp <- center/V",
    },
    "iv_1cmt": {
        "is_iv": True, "struct": ["CL", "V"], "iiv": ["CL", "V"],
        "ode": ["d/dt(center) <- -(CL/V)*center"],
        "cp": "cp <- center/V",
    },
}

_FIT_TIMEOUT_S = 600
_nlmixr2_present: bool | None = None  # cached availability probe
_rosetta: bool | None = None          # cached Rosetta-translation check


def _under_rosetta() -> bool:
    """True when an x86_64 Python is running on an Apple-silicon host, so a child
    ``Rscript`` would inherit the x86_64 preference and compile an x86_64 model
    object that the native arm64 R cannot ``dlopen`` — R must then be forced to
    arm64 (see :func:`_r_cmd`).

    Detection does not rely on ``sysctl.proc_translated``, which is unreliable
    here: from an x86_64 parent the ``sysctl`` child may itself be translated
    (reporting 1), and in a sandbox the call can be blocked entirely. Instead we
    ask the decisive question directly — *can an arm64 binary run at all?* That
    succeeds only on Apple silicon and needs no special permission. The check is
    gated on this process being x86_64, so a native arm64 or a genuine Intel host
    both correctly return False."""
    global _rosetta
    if _rosetta is None:
        _rosetta = False
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            arch = shutil.which("arch")
            if arch:
                try:
                    probe = subprocess.run([arch, "-arm64", "/usr/bin/true"],
                                           capture_output=True, timeout=5)
                    _rosetta = probe.returncode == 0
                except Exception:
                    _rosetta = False
    return _rosetta


def _r_cmd(args: list[str]) -> list[str]:
    """Prefix an Rscript invocation with ``arch -arm64`` when under Rosetta so R
    (and the model compiler) run natively; a no-op elsewhere."""
    if _under_rosetta() and shutil.which("arch"):
        return ["arch", "-arm64", *args]
    return args


def _nlmixr2_installed() -> bool:
    global _nlmixr2_present
    if _nlmixr2_present is None:
        if shutil.which("Rscript") is None:
            _nlmixr2_present = False
        else:
            try:
                out = subprocess.run(
                    _r_cmd(["Rscript", "-e",
                            'cat(requireNamespace("nlmixr2", quietly=TRUE) && '
                            'requireNamespace("jsonlite", quietly=TRUE))']),
                    capture_output=True, text=True, timeout=60,
                )
                _nlmixr2_present = out.stdout.strip().upper().startswith("TRUE")
            except Exception:
                _nlmixr2_present = False
    return _nlmixr2_present


def _build_r_script(model_key: str, iiv_params: list[str], defaults: dict[str, float],
                    est: str, allometric: dict[str, float] | None = None) -> str:
    cfg = _SUPPORTED[model_key]
    allo = allometric or {}
    iiv = [p for p in (iiv_params or cfg["iiv"]) if p in cfg["struct"]] or cfg["iiv"]

    ini_lines = [f"tv{p} <- log({float(defaults[p])})" for p in cfg["struct"]]
    ini_lines += [f"eta.{p} ~ 0.1" for p in iiv]
    ini_lines.append("prop.sd <- 0.1")

    model_lines = []
    for p in cfg["struct"]:
        rhs = f"exp(tv{p} + eta.{p})" if p in iiv else f"exp(tv{p})"
        # Allometric WT scaling, mirroring the native fitter's scale_params, so the
        # estimated tv* are 70-kg-centered and every engine is scored on identical
        # footing (the scoring path re-applies the same (WT/70)^expo).
        expo = float(allo.get(p, 0.0))
        if expo:
            rhs = f"({rhs}) * (WT/70.0)^{expo}"
        model_lines.append(f"{p} <- {rhs}")
    model_lines += cfg["ode"]
    model_lines.append(cfg["cp"])
    model_lines.append("cp ~ prop(prop.sd)")

    param_extract = ", ".join(f'{p}=exp(pf[["tv{p}"]])' for p in cfg["struct"])
    omega_extract = "\n".join(
        f'if ("eta.{p}" %in% rownames(om)) ocv${p} <- cv(om["eta.{p}","eta.{p}"])'
        for p in iiv)

    ind = "  "
    ini_block = "\n".join(ind * 2 + ln for ln in ini_lines)
    model_block = "\n".join(ind * 2 + ln for ln in model_lines)
    return f"""suppressMessages({{library(nlmixr2); library(jsonlite)}})
a <- commandArgs(trailingOnly=TRUE); csv <- a[1]; outjson <- a[2]
d <- read.csv(csv, na.strings="."); d$AMT[is.na(d$AMT)] <- 0
mod <- function() {{
  ini({{
{ini_block}
  }})
  model({{
{model_block}
  }})
}}
fit <- tryCatch(suppressWarnings(nlmixr2(mod, d, est="{est}", control=foceiControl(print=0))),
  error=function(e){{writeLines(toJSON(list(status="failed",message=conditionMessage(e)),auto_unbox=TRUE),outjson);quit(status=0)}})
pf <- fixef(fit); om <- fit$omega; cv <- function(v) sqrt(exp(v)-1)*100
ocv <- list()
{omega_extract}
prop <- tryCatch(unname(pf[["prop.sd"]]), error=function(e) NA)
# best-effort convergence: a completed fit with a finite objective and finite
# fixed effects (guards degenerate/blown-up fits; nlmixr2 exposes no plain flag)
conv <- tryCatch(is.finite(as.numeric(fit$objf)) && all(is.finite(as.numeric(pf))),
                 error=function(e) FALSE)
out <- list(status="ok", params=list({param_extract}),
            omega_cv_pct=ocv, sigma=list(prop=prop, add=NULL),
            objf=as.numeric(fit$objf), converged=conv)
writeLines(toJSON(out, auto_unbox=TRUE, na="null"), outjson)
"""


class Nlmixr2Adapter:
    name = "nlmixr2"

    def available(self) -> bool:
        return _nlmixr2_installed()

    def fit(self, spec: CandidateSpec, subjects: list[dict], *,
            seed: int = 20250614) -> EngineResult:
        if spec.model_key not in _SUPPORTED:
            return EngineResult(engine=self.name, model_name=spec.model_key,
                                status="failed",
                                message=f"nlmixr2 v0 supports {sorted(_SUPPORTED)}")
        from app.compute.pk_models import get_model

        model = get_model(spec.model_key)
        est = "saem" if spec.method.lower() == "saem" else "focei"
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nlmixr2_") as tmp:
            csv = write_nonmem_csv(subjects, os.path.join(tmp, "data.csv"),
                                   is_iv=model.is_iv)
            script = os.path.join(tmp, "fit.R")
            with open(script, "w") as fh:
                fh.write(_build_r_script(spec.model_key, spec.iiv_params or [],
                                         dict(model.defaults), est,
                                         allometric=dict(model.allometric or {})))
            out_json = os.path.join(tmp, "out.json")
            try:
                subprocess.run(_r_cmd(["Rscript", script, csv, out_json]),
                               capture_output=True, text=True, timeout=_FIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return EngineResult(engine=self.name, model_name=spec.model_key,
                                    status="failed", message="nlmixr2 fit timed out")
            if not os.path.exists(out_json):
                return EngineResult(engine=self.name, model_name=spec.model_key,
                                    status="failed", message="nlmixr2 produced no output")
            with open(out_json) as fh:
                res = json.load(fh)
        dt = time.perf_counter() - t0

        if res.get("status") != "ok":
            return EngineResult(engine=self.name, model_name=spec.model_key,
                                status="failed", message=res.get("message", "fit failed"))

        params = {k: float(v) for k, v in res["params"].items()}
        omega_cv = {k: float(v) for k, v in (res.get("omega_cv_pct") or {}).items()}
        sigma_prop = res.get("sigma", {}).get("prop")
        iiv = list(omega_cv) or (spec.iiv_params or _SUPPORTED[spec.model_key]["iiv"])
        sc = score_from_population(
            spec.model_key, subjects, theta=params, omega_cv_pct=omega_cv,
            sigma_prop=sigma_prop, sigma_add=None, iiv_params=iiv,
            error_model="proportional",
        )
        return EngineResult(
            engine=f"nlmixr2_{est}", engine_version="nlmixr2", model_name=spec.model_key,
            converged=bool(res.get("converged")), runtime_s=dt,
            ofv=res.get("objf"),  # within-engine only; not derived to AIC/BIC (external scale)
            params=params, omega_cv_pct=omega_cv,
            sigma={"prop": sigma_prop, "add": None}, iiv_params=iiv,
            error_model="proportional", n_subjects=len(subjects),
            raw=res, **sc,
        )
