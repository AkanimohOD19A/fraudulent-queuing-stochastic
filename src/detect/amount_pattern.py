"""
Detects the "low-value probe, then high-value strike" pattern — e.g. a
5 naira auth check to confirm a stolen card is live, followed shortly by
a near-limit transaction. Thresholds are DERIVED per account from that
account's own transaction history via streaming quantiles (P2Quantile),
not hardcoded amounts — "suspiciously low" for a student's account and
for a corporate account are very different numbers.
"""
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from streaming_quantile import P2Quantile


@dataclass
class AccountProfile:
    low_q: P2Quantile = field(default_factory=lambda: P2Quantile(0.02))
    high_q: P2Quantile = field(default_factory=lambda: P2Quantile(0.98))
    recent: deque = field(default_factory=lambda: deque(maxlen=20))  # (time, amount)
    n_seen: int = 0


class ProbeStrikeDetector:
    def __init__(self, strike_window_sec: float = 300.0, min_history: int = 20):
        """
        strike_window_sec: how long after a low-value probe a high-value
            transaction still counts as a follow-up strike (tune per
            domain — 5 min is a reasonable start for card-testing).
        min_history: minimum transactions on an account before its
            thresholds are trusted enough to flag on (avoids flagging a
            brand-new account's very first small/large transactions).
        """
        self.strike_window_sec = strike_window_sec
        self.min_history = min_history
        self.profiles: dict[str, AccountProfile] = defaultdict(AccountProfile)

    def observe(self, account_id: str, t: float, amount: float) -> dict | None:
        """
        Feed one transaction in. Returns a flag dict if this transaction
        completes a probe->strike pattern, else None. Always updates the
        account's running thresholds (online — no stored history needed
        beyond the small `recent` buffer used to locate the probe).
        """
        profile = self.profiles[account_id]
        flag = None

        if profile.n_seen >= self.min_history:
            low_thresh = profile.low_q.value()
            high_thresh = profile.high_q.value()

            if amount >= high_thresh:
                # look back through recent transactions for a qualifying probe
                for probe_t, probe_amount in reversed(profile.recent):
                    if t - probe_t > self.strike_window_sec:
                        break
                    if probe_amount <= low_thresh:
                        flag = {
                            "account_id": account_id,
                            "probe_time": probe_t,
                            "probe_amount": probe_amount,
                            "strike_time": t,
                            "strike_amount": amount,
                            "low_threshold": low_thresh,
                            "high_threshold": high_thresh,
                            "gap_sec": t - probe_t,
                        }
                        break

        profile.low_q.update(amount)
        profile.high_q.update(amount)
        profile.recent.append((t, amount))
        profile.n_seen += 1
        return flag


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)

    detector = ProbeStrikeDetector(strike_window_sec=120.0, min_history=15)
    account = "acct_001"
    t = 0.0
    flags = []

    # normal history: typical purchase amounts
    for _ in range(30):
        t += rng.exponential(20)
        amount = rng.lognormal(mean=4.0, sigma=0.6)
        result = detector.observe(account, t, amount)
        if result:
            flags.append(result)

    # attack: tiny probe, then a near-limit strike shortly after
    t += 5
    detector.observe(account, t, 5.0)  # probe
    t += 40
    result = detector.observe(account, t, 999999.0)  # strike
    if result:
        flags.append(result)

    for f in flags:
        print(
            f"FLAGGED {f['account_id']}: probe {f['probe_amount']:.2f} at "
            f"t={f['probe_time']:.0f}, strike {f['strike_amount']:.2f} at "
            f"t={f['strike_time']:.0f} (gap {f['gap_sec']:.0f}s, "
            f"thresholds low={f['low_threshold']:.2f} high={f['high_threshold']:.2f})"
        )
    print(f"Total flags: {len(flags)}")
