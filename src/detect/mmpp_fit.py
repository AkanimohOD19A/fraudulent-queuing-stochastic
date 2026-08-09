"""
Fit a 2-state MMPP-style detector by binning event counts and fitting a
Gaussian HMM over the binned counts (a standard, easy-to-implement
approximation to a true Poisson-HMM/MMPP when hmmlearn's PoissonHMM isn't
available). Viterbi-decode the most likely state sequence; state 1 (the
higher-rate state) becomes the flagged/attack windows.
"""
from __future__ import annotations
import numpy as np
from hmmlearn.hmm import GaussianHMM


def bin_events(event_times: np.ndarray, T: float, dt: float = 1.0) -> np.ndarray:
    """Return per-bin event counts as a (n_bins, 1) array (hmmlearn expects 2D)."""
    n_bins = int(np.ceil(T / dt))
    counts, _ = np.histogram(event_times, bins=n_bins, range=(0, T))
    return counts.reshape(-1, 1).astype(float)


def fit_mmpp(event_times: np.ndarray, T: float, dt: float = 1.0, n_states: int = 2,
             random_state: int = 0) -> dict:
    counts = bin_events(event_times, T, dt)

    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=200, random_state=random_state)
    model.fit(counts)

    log_likelihood = model.score(counts)
    states = model.predict(counts)  # Viterbi decode

    # identify the "attack" state as whichever has the higher mean rate
    means = model.means_.flatten()
    attack_state = int(np.argmax(means))
    flags = states == attack_state

    n_params = n_states * n_states + n_states * 2  # transition + (mean, var) per state, roughly
    n_obs = len(counts)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(max(n_obs, 1)) - 2 * log_likelihood

    grid = np.arange(0, T, dt)[: len(states)]

    return {
        "means": means, "attack_state": attack_state,
        "log_likelihood": log_likelihood, "aic": aic, "bic": bic,
        "grid": grid, "states": states, "flags": flags,
        "counts": counts.flatten(),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
    from arrivals import mmpp_stream

    rng = np.random.default_rng(3)
    events, labels = mmpp_stream(
        mu_normal=1.0, mu_attack=8.0,
        p_normal_to_attack=0.02, p_attack_to_normal=0.15,
        duration=300.0, rng=rng,
    )
    result = fit_mmpp(events, T=300.0)
    print(f"State means (events/bin): {result['means']}")
    print(f"Attack state index: {result['attack_state']}")
    print(f"Log-likelihood: {result['log_likelihood']:.2f}  AIC: {result['aic']:.2f}  BIC: {result['bic']:.2f}")
    print(f"Flagged bins: {result['flags'].sum()} / {len(result['flags'])}")
