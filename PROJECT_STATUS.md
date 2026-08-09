# Project status — fraud-queueing-stochastic

A living checklist. Update the checkboxes as you wire things together —
this is meant to be the one place that answers "what state is this
project actually in?"

Three questions for every component:
1. **Built** — does it run and pass its own standalone test?
2. **Wired** — is it actually connected to the dashboard or another
   running piece, or does it just sit there working in isolation?
3. **Live creds** — is it using a real API key/service, or a graceful
   fallback placeholder?

## Core simulation + detection (the dashboard's spine)

| Component | Built | Wired into dashboard | Needs live creds |
|---|---|---|---|
| `simulate/arrivals.py` (Poisson/Hawkes/MMPP generators) | ✅ | ✅ | no |
| `simulate/queue_model.py` (M/M/c queue) | ✅ | ✅ | no |
| `detect/hawkes_fit.py` | ✅ | ✅ | no |
| `detect/mmpp_fit.py` | ✅ | ✅ | no |
| `eval/harness.py` (comparison table) | ✅ | ✅ | no |
| `explain/cohere_summary.py` | ✅ | ✅ (fallback text — no key set yet) | yes, for real LLM output |
| `trace/langfuse_setup.py` | ✅ | ✅ (now wraps all 5 dashboard stages as of this update) | yes — set keys in `.env`, dashboard now calls `load_dotenv()` |

## Built and tested, but standing alone (not on the assembly line yet)

| Component | Built | Wired anywhere | Needs live creds |
|---|---|---|---|
| `detect/streaming_quantile.py` (P^2 estimator) | ✅ | ❌ | no |
| `detect/amount_pattern.py` (probe-strike detector) | ✅ | ❌ | no |
| `eval/mlflow_tracking.py` | ✅ | ❌ (run it manually, separate from dashboard) | no (local sqlite) |
| `eval/ci_gate.py` | ✅ | ❌ locally — only runs inside GitHub Actions | no |

## Orchestration (the "concrete production ML" layer)

| Component | Built | Wired | Needs live creds |
|---|---|---|---|
| `orchestrate/pipeline.py` (ZenML: ingest -> train -> evaluate/track -> quality gate) | ✅ ran end-to-end in testing | ✅ — probe-strike detector now actually called from here | no |
| Real-time replay / live monitoring view | ❌ not built yet | — | — |

## Infrastructure

| Component | Status |
|---|---|
| `.github/workflows/eval-and-track.yml` | Written, tested logic locally — but **not yet inert-proof**: it only actually runs once this project is pushed to a real GitHub repo. Right now it's a file sitting in a zip on your machine. |
| Open-source repo (public GitHub) | Not yet created — this is the single biggest missing link. Nothing here is "open source" in practice until it has a public repo URL. |

## Suggested next single step

Don't try to wire everything at once. Pick one:
- **Push to GitHub first** — this is what unblocks the CI/CD story entirely
  (the workflow file is currently inert without it), and it's what makes
  the project genuinely "open source" rather than just locally correct.
- **Or wire the probe-strike detector into the dashboard** — smaller,
  self-contained, good warm-up before tackling GitHub.

Either is reasonable — just pick one and finish it before starting the
next, rather than holding all five open threads at once.
