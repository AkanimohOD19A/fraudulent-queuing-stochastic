"""
Simple CI quality gate: re-runs the tracked comparison and fails (exit 1)
if either model's F1 on its own generative stream drops below a floor.
This is the piece that turns MLflow tracking into an actual CI/CD gate
rather than just a logging exercise — a regression in detection quality
blocks the merge instead of shipping silently.
"""
from __future__ import annotations
import sys
from mlflow_tracking import run_comparison_with_tracking

MIN_F1 = {"hawkes": 0.15, "mmpp": 0.85}  # tune these once you have a real baseline


def check(stream_type: str, model_col: str) -> bool:
    df = run_comparison_with_tracking(stream_type, T=300.0, seed=11)
    row = df[df["model"] == model_col].iloc[0]
    f1 = row[f"{model_col}_f1"]
    passed = f1 >= MIN_F1[model_col]
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {stream_type} stream / {model_col}: F1={f1:.3f} (floor {MIN_F1[model_col]})")
    return passed


if __name__ == "__main__":
    results = [
        check("hawkes", "hawkes"),
        check("mmpp", "mmpp"),
    ]
    if not all(results):
        print("\nQuality gate failed — a detector regressed below its floor.")
        sys.exit(1)
    print("\nAll detectors meet their quality floor.")
