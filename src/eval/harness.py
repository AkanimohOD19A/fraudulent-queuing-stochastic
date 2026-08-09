"""
Evaluation harness: runs both the Hawkes and MMPP detectors against a
labeled event stream and produces a single comparison report.

Designed to be reusable — swap in other detectors later (this is the same
rubric-and-trace pattern worth reusing for the agent-evaluation project).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd


def _events_to_bin_labels(event_times: np.ndarray, event_labels: np.ndarray,
                           T: float, dt: float) -> np.ndarray:
    """Collapse per-event ground-truth labels into per-bin ground truth
    (a bin counts as 'attack' if any event in it was labeled attack)."""
    n_bins = int(np.ceil(T / dt))
    grid_labels = np.zeros(n_bins, dtype=int)
    bin_idx = np.minimum((event_times // dt).astype(int), n_bins - 1)
    for idx, lbl in zip(bin_idx, event_labels):
        if lbl == 1:
            grid_labels[idx] = 1
    return grid_labels


def _precision_recall_f1(flags: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float, float]:
    flags = flags.astype(bool)
    gt = ground_truth.astype(bool)
    tp = np.sum(flags & gt)
    fp = np.sum(flags & ~gt)
    fn = np.sum(~flags & gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _detection_latency(flags: np.ndarray, ground_truth: np.ndarray, grid: np.ndarray) -> float | None:
    """Seconds between the first ground-truth attack bin and the first
    flagged bin at or after it, for the first contiguous attack episode.
    Returns None if the attack was never flagged."""
    gt_idx = np.where(ground_truth == 1)[0]
    if len(gt_idx) == 0:
        return None
    attack_start_idx = gt_idx[0]
    flagged_after = np.where(flags[attack_start_idx:])[0]
    if len(flagged_after) == 0:
        return None
    detected_idx = attack_start_idx + flagged_after[0]
    return float(grid[detected_idx] - grid[attack_start_idx])


def evaluate_detector(name: str, fit_fn, event_times: np.ndarray, event_labels: np.ndarray,
                       T: float, dt: float = 1.0, **fit_kwargs) -> dict:
    """
    fit_fn: a function like flag_windows() or fit_mmpp() that takes
    (event_times, T, ...) and returns a dict containing at least
    'grid', 'flags', 'log_likelihood', 'aic', 'bic'.
    """
    ground_truth = _events_to_bin_labels(event_times, event_labels, T, dt)

    start = time.perf_counter()
    result = fit_fn(event_times, T=T, **fit_kwargs)
    fit_time = time.perf_counter() - start

    grid = result["grid"]
    flags = result["flags"][: len(ground_truth)]
    gt_aligned = ground_truth[: len(flags)]

    precision, recall, f1 = _precision_recall_f1(flags, gt_aligned)
    latency = _detection_latency(flags, gt_aligned, grid[: len(flags)])

    return {
        "model": name,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "detection_latency_sec": latency,
        "log_likelihood": round(result.get("log_likelihood", float("nan")), 2),
        "aic": round(result.get("aic", float("nan")), 2),
        "bic": round(result.get("bic", float("nan")), 2),
        "fit_time_sec": round(fit_time, 4),
    }


def compare_models(event_times: np.ndarray, event_labels: np.ndarray, T: float,
                    dt: float = 1.0) -> pd.DataFrame:
    """Runs both detectors on the same stream and returns a comparison table."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detect"))
    from hawkes_fit import flag_windows
    from mmpp_fit import fit_mmpp

    rows = [
        evaluate_detector("Hawkes", flag_windows, event_times, event_labels, T, dt),
        evaluate_detector("MMPP", fit_mmpp, event_times, event_labels, T, dt),
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
    from arrivals import hawkes_stream, mmpp_stream

    rng = np.random.default_rng(11)

    print("=== Cross-test: Hawkes-generated stream ===")
    ev, lbl = hawkes_stream(mu=0.5, alpha=0.3, beta=1.0, duration=300.0, rng=rng,
                             attack_windows=[(120.0, 150.0)], attack_alpha_multiplier=2.5)
    print(compare_models(ev, lbl, T=300.0).to_string(index=False))

    print("\n=== Cross-test: MMPP-generated stream ===")
    ev2, lbl2 = mmpp_stream(mu_normal=1.0, mu_attack=8.0, p_normal_to_attack=0.02,
                             p_attack_to_normal=0.15, duration=300.0, rng=rng)
    print(compare_models(ev2, lbl2, T=300.0).to_string(index=False))
