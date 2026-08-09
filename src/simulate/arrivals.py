"""
Arrival-process generators for the fraud-queueing project.

Three generators, all returning a sorted 1D numpy array of event
timestamps (in seconds, starting at 0) plus a ground-truth label array
(0 = normal, 1 = attack) at whatever bin resolution you request.

- poisson_baseline: homogeneous Poisson process (legitimate traffic)
- hawkes_stream:    self-exciting process (card-testing-style bursts)
- mmpp_stream:      2-state Markov-modulated Poisson process
"""
from __future__ import annotations
import numpy as np


def poisson_baseline(rate: float, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Homogeneous Poisson process via exponential inter-arrival times."""
    events = []
    t = 0.0
    while t < duration:
        t += rng.exponential(1.0 / rate)
        if t < duration:
            events.append(t)
    return np.array(events)


def hawkes_stream(
    mu: float,
    alpha: float,
    beta: float,
    duration: float,
    rng: np.random.Generator,
    attack_windows: list[tuple[float, float]] | None = None,
    attack_alpha_multiplier: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hawkes process via Ogata's thinning algorithm.

    Conditional intensity: lambda(t) = mu + sum_i alpha * exp(-beta * (t - t_i))
    for all past events t_i < t.

    `attack_windows` optionally boosts alpha (self-excitation strength)
    inside given [start, end) windows to simulate a card-testing burst
    riding on top of otherwise-normal Hawkes traffic. This gives you a
    stream with a genuine ground-truth attack label even though the whole
    process is Hawkes-generated.

    Returns (event_times, ground_truth_labels) where ground_truth_labels
    is 1 for events that fall inside an attack window, 0 otherwise.
    """
    attack_windows = attack_windows or []

    def alpha_at(t: float) -> float:
        for start, end in attack_windows:
            if start <= t < end:
                return alpha * attack_alpha_multiplier
        return alpha

    events: list[float] = []
    t = 0.0
    # upper bound on intensity for thinning; recompute conservatively
    while t < duration:
        # current intensity estimate (upper bound: assume max alpha regime)
        cur_alpha = max(alpha, alpha * attack_alpha_multiplier)
        lam_upper = mu + sum(
            cur_alpha * np.exp(-beta * (t - te)) for te in events[-200:]  # cap for speed
        ) + cur_alpha  # small slack

        t += rng.exponential(1.0 / max(lam_upper, 1e-6))
        if t >= duration:
            break

        # true intensity at candidate time t
        a_t = alpha_at(t)
        lam_true = mu + sum(
            a_t * np.exp(-beta * (t - te)) for te in events[-200:]
        )
        if rng.uniform(0, 1) <= lam_true / lam_upper:
            events.append(t)

    events = np.array(events)
    labels = np.array(
        [1 if any(s <= e < end for s, end in attack_windows) else 0 for e in events],
        dtype=int,
    )
    return events, labels


def mmpp_stream(
    mu_normal: float,
    mu_attack: float,
    p_normal_to_attack: float,
    p_attack_to_normal: float,
    duration: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2-state Markov-Modulated Poisson Process.

    A discrete-time Markov chain (state resolution `dt` seconds) switches
    between 'normal' and 'attack' states; within each dt-bin, events are
    drawn from a Poisson process at that state's rate.

    Returns (event_times, ground_truth_labels_per_event).
    """
    n_bins = int(duration / dt)
    state = 0  # 0 = normal, 1 = attack
    events: list[float] = []
    labels: list[int] = []

    for i in range(n_bins):
        bin_start = i * dt
        rate = mu_attack if state == 1 else mu_normal
        n_events = rng.poisson(rate * dt)
        if n_events > 0:
            local_times = np.sort(rng.uniform(0, dt, size=n_events)) + bin_start
            events.extend(local_times.tolist())
            labels.extend([state] * n_events)

        # transition
        if state == 0 and rng.uniform(0, 1) < p_normal_to_attack:
            state = 1
        elif state == 1 and rng.uniform(0, 1) < p_attack_to_normal:
            state = 0

    return np.array(events), np.array(labels, dtype=int)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    base = poisson_baseline(rate=2.0, duration=60.0, rng=rng)
    print(f"Poisson baseline: {len(base)} events in 60s")

    # Note: keep alpha/beta (the branching ratio) comfortably below 1,
    # even at the attack multiplier, or the process is supercritical and
    # explodes (unbounded event count). 0.3*2.5/1.0 = 0.75 stays subcritical.
    hawkes_ev, hawkes_lbl = hawkes_stream(
        mu=0.5, alpha=0.3, beta=1.0, duration=120.0, rng=rng,
        attack_windows=[(40.0, 55.0)], attack_alpha_multiplier=2.5,
    )
    print(f"Hawkes stream: {len(hawkes_ev)} events, {hawkes_lbl.sum()} in attack window")

    mmpp_ev, mmpp_lbl = mmpp_stream(
        mu_normal=1.0, mu_attack=8.0,
        p_normal_to_attack=0.03, p_attack_to_normal=0.15,
        duration=300.0, rng=rng,
    )
    print(f"MMPP stream: {len(mmpp_ev)} events, {mmpp_lbl.sum()} in attack state")
