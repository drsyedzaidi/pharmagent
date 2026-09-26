# PharmAgent

**Agentic pharmacometrics — an AI assistant that *runs* validated PK/PD analyses instead of free-handing the numbers.**

The design thesis is *agents decide, tools execute*: a language model plans the analysis, but every reported number comes from a validated, deterministic compute tool with a tamper-evident audit trail — never from tokens. This repo is the platform (FastAPI backend + React frontend) plus **PharmacometricsBench**, an eval that measures exactly that difference.

<p align="center"><img src="papers/figure_tool_fidelity.svg" width="720" alt="Tool fidelity: tool-grounded 1.00, Opus 4.8 alone 0.93, Haiku 4.5 alone 0.60"></p>

> **The headline result** (mean of 3 runs, 30 tasks, 5 categories): a tool-grounded agent scores **1.00** — exact by construction. A frontier model reasoning to the numbers unaided (Claude Opus 4.8) scores **0.93 ± 0.03**, near-perfect on formula-driven tasks but only **0.74** on iterative compartmental fitting; a smaller model (Haiku 4.5) scores **0.60 ± 0.04**. Scale narrows the gap; only the tools close it. Full data: [`papers/`](papers/).

---

## What's in here

| Path | What it is |
|---|---|
| [`backend/`](backend/) | FastAPI service — deterministic compute, NLME solvers, cross-engine orchestration, audit chain, SQLite persistence |
| [`backend/pharmacometricsbench/`](backend/pharmacometricsbench/) | The eval: tool-grounded ground truth, keyless MockLLM + real-model reference agents, PK-DB real-data loader |
| [`frontend/`](frontend/) | React + Vite + TypeScript UI |
| [`dashboard/`](dashboard/) | Marketing/landing dashboard (Vite + Netlify) |
| [`papers/`](papers/) | Tool-fidelity figure, caption, and pinned reproducible run logs |
| [`PHARMACOMETRICSBENCH.md`](PHARMACOMETRICSBENCH.md) · [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Design docs |

## Capabilities (backend)

- **Deterministic PK compute** — NCA (incl. steady-state), bioequivalence, dose-proportionality, compartmental fitting, exposure simulation, VPC / pcVPC, GOF diagnostics, dose sweeps.
- **True NLME** — FOCE-I and SAEM estimation, covariate modelling, stepwise covariate selection (SCM), MAP/TDM Bayesian forecasting, BLQ/M3 censored likelihood.
- **Cross-engine orchestration** — run candidate models across estimation engines and rank on *engine-agnostic prediction accuracy* (not incomparable native OFV/AIC/BIC), including a real nlmixr2 (R) adapter.
- **Provenance & governance** — SHA-256 audit hash-chain, human-in-the-loop review gate, run reproducibility reports, exports (CSV / DOCX / CDISC-ADaM / NONMEM `.ctl` / mrgsolve `.cpp`).

## PharmacometricsBench

A reproducible eval for whether an agent can *run* a pharmacometric analysis correctly. Ground truth for each task is the output a validated compute tool produces on the provided data, so the score measures reproduction, not eyeballing.

```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# keyless (MockLLM) reference agents — no API key needed
python -m pharmacometricsbench.runner --run oracle naive

# real model rows (needs ANTHROPIC_API_KEY); model-pinned reference agents
python -m pharmacometricsbench.runner --run oracle naive llm-opus llm-haiku
```

Real-drug FIH answer keys are harvested from [PK-DB](https://pk-db.com) with one command (writes zero fabricated tasks without a token):

```bash
python -m pharmacometricsbench.pkdb.build_fih_taskset            # anonymous → 0 tasks (auth-gated)
PKDB_API_TOKEN=... python -m pharmacometricsbench.pkdb.build_fih_taskset   # real cited tasks
```

## Quickstart

**Backend**
```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # no --reload; restart after edits
```

**Frontend**
```bash
cd frontend && npm ci && npm run dev
```

The app runs fully **keyless** by default — `MockLLM` (deterministic) stands in for a real model, and the entire test suite is keyless. Set `ANTHROPIC_API_KEY` (see [`backend/.env.example`](backend/.env.example)) to use a real model.

## LLM providers — Claude, free/local (Ollama), or keyless mock

Everything quantitative is a deterministic tool; the LLM only routes a request
to an agent and picks the next tool. So a small local model is enough, and no
data leaves the machine.

| provider | set | notes |
|---|---|---|
| Claude (Anthropic) | `PHARMAGENT_ANTHROPIC_API_KEY=sk-ant-…` (+ `PHARMAGENT_MODEL`) | best routing / argument composition |
| **Local, free (Ollama)** | `PHARMAGENT_LLM_PROVIDER=openai` `PHARMAGENT_MODEL=qwen2.5:7b` | `ollama pull qwen2.5:7b` first; default URL `http://127.0.0.1:11434/v1`, no key |
| LM Studio / vLLM / any OpenAI-compatible server | `PHARMAGENT_LLM_BASE_URL=http://127.0.0.1:1234/v1` `PHARMAGENT_MODEL=<loaded model>` | model must support tool calling |
| Hosted free tiers (OpenRouter, Groq) | `PHARMAGENT_LLM_BASE_URL=https://openrouter.ai/api/v1` `PHARMAGENT_LLM_API_KEY=…` `PHARMAGENT_MODEL=…` | data leaves the machine |
| Mock (default) | nothing | deterministic keyword routing, one tool per agent, never confirms expensive runs |

`PHARMAGENT_LLM_PROVIDER` is `auto` by default: Anthropic key → Claude; else a
base URL → OpenAI-compatible; else mock. Force one with `anthropic` / `openai`
/ `mock`. `/api/health` reports the active provider and the UI badge shows it.
Models that only answer in prose (no tool calling) simply pick no tool.

**Switch at runtime from the UI**: click the "Backend online · …" badge → pick
Mock / Local (Ollama, lists the models you have pulled) / ChatGPT (OpenAI key) /
Claude (Anthropic key) → *Test* → *Use this model*. The choice applies to every
session immediately and is remembered (provider, model, URL) in
`<data_dir>/llm_settings.json`; API keys are held in the backend's memory only
and must be re-entered after a relaunch unless set in the environment. The same
switch is `GET/PUT /api/llm` (bearer-token gated when `PHARMAGENT_API_TOKEN` is set).

## Desktop app (macOS)

One process, one window: the FastAPI backend serves the built React frontend
from the same loopback origin and a native WebKit window (pywebview) opens on
it. No Node, no Vite, no browser tab. Per-user data (SQLite, uploads, reports)
lives in `~/Library/Application Support/PharmAgent/`, never inside the bundle.

```bash
cd backend && .venv/bin/pip install -r requirements-desktop.txt   # pywebview + pyinstaller (once)
./desktop/build.sh --smoke        # builds frontend, bundles, launches headless, hits /api/health
open desktop/dist/PharmAgent.app
```

The bundle is unsigned: on first launch right-click → Open (or
`xattr -dr com.apple.quarantine desktop/dist/PharmAgent.app`). Run from source
without bundling: `backend/.venv/bin/python desktop/pharmagent_desktop.py`
(needs `npm --prefix frontend run build` first). Env knobs:
`PHARMAGENT_DESKTOP_PORT`, `PHARMAGENT_DESKTOP_DATA_DIR`,
`PHARMAGENT_DESKTOP_NO_WINDOW=1` (serve only). A real LLM needs
`PHARMAGENT_ANTHROPIC_API_KEY` in the environment the app is launched from;
otherwise it runs on the deterministic MockLLM. The nlmixr2 cross-engine
comparison needs `Rscript` on the launch PATH and is simply unavailable
otherwise.

## Tests & CI

```bash
cd backend && pytest -q            # ~300 tests
ruff check app tests               # lint
```

CI (`.github/workflows/ci.yml`) runs backend lint + tests and frontend typecheck + build on every push.

---

*Research/engineering project in clinical pharmacology & pharmacometrics. Not for clinical use.*
