"""
ZenML pipeline tying together the pieces you already have, tested, and
working as standalone modules. This is the orchestration layer — it
doesn't reimplement anything, it just calls your existing functions in
the right order and lets ZenML track each run, its inputs/outputs, and
its lineage automatically.

Run with:  python src/orchestrate/pipeline.py
"""
from __future__ import annotations
import os
import sys
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step, pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eval"))

from arrivals import hawkes_stream, mmpp_stream
from amount_pattern import ProbeStrikeDetector
from hawkes_fit import flag_windows
from mmpp_fit import fit_mmpp
from harness import evaluate_detector


@step
def ingest_step(stream_type: str, duration: float, seed: int) -> Annotated[pd.DataFrame, "transactions"]:
    """Phase 1: generate a realistic (synthetic) transaction stream with
    embedded attack windows, PLUS run the probe-strike (low-then-high
    amount) detector inline as each transaction 'arrives' — this is the
    piece that handles the small-then-suspiciously-large pattern."""
    rng = np.random.default_rng(seed)

    if stream_type == "hawkes":
        events, labels = hawkes_stream(
            mu=0.5, alpha=0.3, beta=1.0, duration=duration, rng=rng,
            attack_windows=[(duration * 0.4, duration * 0.5)], attack_alpha_multiplier=2.5,
        )
    else:
        events, labels = mmpp_stream(
            mu_normal=1.0, mu_attack=8.0, p_normal_to_attack=0.02,
            p_attack_to_normal=0.15, duration=duration, rng=rng,
        )

    # assign amounts: attack-labeled events skew toward the probe/strike
    # pattern, normal events get an ordinary lognormal spread
    amounts = np.where(
        labels == 1,
        rng.choice([5.0, 999999.0], size=len(events), p=[0.5, 0.5]),
        rng.lognormal(mean=4.0, sigma=0.6, size=len(events)),
    )
    account_ids = rng.choice([f"acct_{i:03d}" for i in range(5)], size=len(events))

    df = pd.DataFrame({
        "time": events, "amount": amounts, "account_id": account_ids, "label": labels,
    })

    # run the probe-strike detector as each row "arrives", in time order
    detector = ProbeStrikeDetector(strike_window_sec=120.0, min_history=10)
    probe_flags = []
    for row in df.sort_values("time").itertuples():
        flag = detector.observe(row.account_id, row.time, row.amount)
        if flag:
            probe_flags.append(flag)
    df.attrs["probe_strike_flags"] = probe_flags

    return df


@step
def train_step(transactions: pd.DataFrame, duration: float) -> Annotated[dict, "fit_results"]:
    """Phase 2: fit both statistical models to the SAME event stream —
    this is the 'model training' step, i.e. parameter estimation via
    MLE (Hawkes) and Baum-Welch/EM (MMPP)."""
    events = transactions["time"].values
    labels = transactions["label"].values

    hawkes_eval = evaluate_detector("hawkes", flag_windows, events, labels, duration)
    mmpp_eval = evaluate_detector("mmpp", fit_mmpp, events, labels, duration)

    return {"hawkes": hawkes_eval, "mmpp": mmpp_eval,
            "n_probe_strike_flags": len(transactions.attrs.get("probe_strike_flags", []))}


@step
def evaluate_and_track_step(fit_results: dict) -> Annotated[pd.DataFrame, "comparison"]:
    """Phase 3 (part 1): score both models side by side and log the run
    to MLflow — params, metrics, and this comparison table as an artifact."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-queueing-stochastic")

    rows = [fit_results["hawkes"], fit_results["mmpp"]]
    df = pd.DataFrame(rows)

    with mlflow.start_run(run_name="zenml_orchestrated_run"):
        for row in rows:
            mlflow.log_metrics({
                f"{row['model']}_precision": row["precision"],
                f"{row['model']}_recall": row["recall"],
                f"{row['model']}_f1": row["f1"],
            })
        mlflow.log_metric("probe_strike_flags", fit_results["n_probe_strike_flags"])
        report_path = "comparison_report.csv"
        df.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

    return df


@step
def quality_gate_step(comparison: pd.DataFrame) -> None:
    """Phase 3 (part 2): the same regression gate used in CI — fails the
    pipeline run itself (not just a separate script) if either model
    drops below its floor. This is what makes 'monitor' real rather than
    just 'log and hope someone looks.'"""
    floors = {"hawkes": 0.10, "mmpp": 0.80}
    for _, row in comparison.iterrows():
        floor = floors.get(row["model"], 0.0)
        if row["f1"] < floor:
            raise ValueError(
                f"Quality gate failed: {row['model']} F1={row['f1']:.3f} below floor {floor}"
            )
    print("Quality gate passed for all models.")


@pipeline
def fraud_detection_pipeline(stream_type: str = "mmpp", duration: float = 300.0, seed: int = 11):
    transactions = ingest_step(stream_type=stream_type, duration=duration, seed=seed)
    fit_results = train_step(transactions=transactions, duration=duration)
    comparison = evaluate_and_track_step(fit_results=fit_results)
    quality_gate_step(comparison=comparison)


if __name__ == "__main__":
    fraud_detection_pipeline()
