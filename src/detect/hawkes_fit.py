"""
Fit an exponential-kernel Hawkes process to observed event times via MLE,
then flag windows where the fitted conditional intensity exceeds a
threshold multiple of the baseline rate mu.

This module treats the model as a *detector*: given only event timestamps
(no ground truth), it estimates mu/alpha/beta and decides, in a rolling
fashion, whether the process currently "looks like" it's in a
self-excited burst.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def hawkes_log_likelihood(params: np.ndarray, event_times: np.ndarray, T: float) -> float:
    """Negative log-likelihood of an exponential-kernel Hawkes process
    (returns negative so it can be minimized)."""
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e10  # keep optimizer in the stable region (alpha < beta => subcritical-ish)

    n = len(event_times)
    if n == 0:
        return mu * T

    # compensator term: integral of intensity over [0, T]
    compensator = mu * T
    # recursive intensity trick: R_i = sum_{j<i} exp(-beta*(t_i - t_j))
    R = 0.0
    log_lik = 0.0
    prev_t = 0.0
    for i, t in enumerate(event_times):
        if i > 0:
            R = np.exp(-beta * (t - prev_t)) * (1 + R)
        lam_i = mu + alpha * R
        log_lik += np.log(max(lam_i, 1e-12))
        prev_t = t

    for t in event_times:
        compensator += (alpha / beta) * (1 - np.exp(-beta * (T - t)))

    return -(log_lik - compensator)


def fit_hawkes(event_times: np.ndarray, T: float) -> dict:
    """Returns fitted mu, alpha, beta plus log-likelihood and AIC/BIC."""
    x0 = np.array([len(event_times) / max(T, 1e-6) * 0.5, 0.2, 1.0])
    res = minimize(
        hawkes_log_likelihood, x0, args=(event_times, T),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    mu, alpha, beta = res.x
    neg_ll = res.fun
    k = 3  # number of params
    n = len(event_times)
    aic = 2 * k + 2 * neg_ll
    bic = k * np.log(max(n, 1)) + 2 * neg_ll
    return {"mu": mu, "alpha": alpha, "beta": beta,
            "log_likelihood": -neg_ll, "aic": aic, "bic": bic}


def rolling_intensity(event_times: np.ndarray, mu: float, alpha: float, beta: float,
                       grid: np.ndarray) -> np.ndarray:
    """Evaluate the fitted conditional intensity lambda(t) at each point in `grid`."""
    intensities = np.zeros_like(grid)
    for i, t in enumerate(grid):
        past = event_times[event_times < t]
        intensities[i] = mu + alpha * np.sum(np.exp(-beta * (t - past)))
    return intensities


def flag_windows(event_times: np.ndarray, T: float, threshold_multiplier: float = 3.0,
                  grid_dt: float = 1.0) -> dict:
    """
    Fit the model, then flag time bins where lambda(t) > threshold_multiplier * mu.
    Returns fit params plus a boolean flag array aligned to `grid`.
    """
    fit = fit_hawkes(event_times, T)
    grid = np.arange(0, T, grid_dt)
    intensities = rolling_intensity(event_times, fit["mu"], fit["alpha"], fit["beta"], grid)
    flags = intensities > threshold_multiplier * fit["mu"]
    return {**fit, "grid": grid, "intensities": intensities, "flags": flags}


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
    from arrivals import hawkes_stream

    rng = np.random.default_rng(7)
    events, labels = hawkes_stream(
        mu=0.5, alpha=0.3, beta=1.0, duration=200.0, rng=rng,
        attack_windows=[(80.0, 100.0)], attack_alpha_multiplier=2.5,
    )
    result = flag_windows(events, T=200.0)
    print(f"Fitted: mu={result['mu']:.3f} alpha={result['alpha']:.3f} beta={result['beta']:.3f}")
    print(f"AIC={result['aic']:.2f} BIC={result['bic']:.2f}")
    flagged_times = result["grid"][result["flags"]]
    print(f"Flagged {len(flagged_times)} bins, "
          f"range {flagged_times.min():.1f}-{flagged_times.max():.1f}" if len(flagged_times) else "No bins flagged")
