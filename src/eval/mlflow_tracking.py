"""
Wraps the eval harness in MLflow tracking: each run logs the generating
stream's parameters (lineage — what data produced this comparison), both
models' fitted params and metrics, and the comparison artifacts (report
CSV, chart). Uses a local file-based tracking store by default (`./mlruns`)
so it works with zero external infra — point MLFLOW_TRACKING_URI at a
real server later without changing this code.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detect"))

import mlflow
import numpy as np
import pandas as pd
from arrivals import hawkes_stream, mmpp_stream
from hawkes_fit import flag_windows
from mmpp_fit import fit_mmpp
from harness import evaluate_detector


def run_comparison_with_tracking(stream_type: str, T: float, seed: int) -> pd.DataFrame:
    # Recent MLflow versions require a database backend rather than the
    # legacy file store — sqlite works with zero external infra, which
    # keeps this lean-resources friendly. Point MLFLOW_TRACKING_URI at a
    # real Postgres/remote server later without touching this code.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-queueing-stochastic")

    rng = np.random.default_rng(seed)

    with mlflow.start_run(run_name=f"{stream_type}_T{int(T)}_seed{seed}"):
        # --- lineage: log exactly what generated this run's data ---
        gen_params = {"stream_type": stream_type, "duration_sec": T, "seed": seed}
        if stream_type == "hawkes":
            gen_params.update(mu=0.5, alpha=0.3, beta=1.0,
                               attack_window="[0.4T, 0.5T]", attack_alpha_multiplier=2.5)
            events, labels = hawkes_stream(
                mu=0.5, alpha=0.3, beta=1.0, duration=T, rng=rng,
                attack_windows=[(T * 0.4, T * 0.5)], attack_alpha_multiplier=2.5,
            )
        else:
            gen_params.update(mu_normal=1.0, mu_attack=8.0,
                               p_normal_to_attack=0.02, p_attack_to_normal=0.15)
            events, labels = mmpp_stream(
                mu_normal=1.0, mu_attack=8.0, p_normal_to_attack=0.02,
                p_attack_to_normal=0.15, duration=T, rng=rng,
            )
        mlflow.log_params({f"gen_{k}": v for k, v in gen_params.items()})
        mlflow.log_dict(gen_params, "lineage/generator_params.json")

        rows = []
        for name, fit_fn in [("hawkes", flag_windows), ("mmpp", fit_mmpp)]:
            result = fit_fn(events, T=T)
            eval_row = evaluate_detector(name, fit_fn, events, labels, T)
            metrics = {
                f"{name}_log_likelihood": result.get("log_likelihood", float("nan")),
                f"{name}_aic": result.get("aic", float("nan")),
                f"{name}_bic": result.get("bic", float("nan")),
                f"{name}_flagged_bins": int(result["flags"].sum()),
                f"{name}_precision": eval_row["precision"],
                f"{name}_recall": eval_row["recall"],
                f"{name}_f1": eval_row["f1"],
            }
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            rows.append({"model": name, **metrics})

            # log fitted params separately per model (namespaced to avoid collisions)
            if name == "hawkes":
                mlflow.log_params({"hawkes_fit_mu": result["mu"],
                                    "hawkes_fit_alpha": result["alpha"],
                                    "hawkes_fit_beta": result["beta"]})
            else:
                mlflow.log_param("mmpp_fit_means", json.dumps(result["means"].tolist()))

        comparison_df = pd.DataFrame(rows)
        report_path = "comparison_report.csv"
        comparison_df.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path, artifact_path="eval")
        os.remove(report_path)

        return comparison_df


if __name__ == "__main__":
    df_hawkes = run_comparison_with_tracking("hawkes", T=300.0, seed=11)
    df_mmpp = run_comparison_with_tracking("mmpp", T=300.0, seed=11)
    print(df_hawkes.to_string(index=False))
    print(df_mmpp.to_string(index=False))
    print("\nRun `mlflow ui --backend-store-uri sqlite:///mlflow.db` to view these runs.")
