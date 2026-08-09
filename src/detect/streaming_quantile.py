"""
P^2 (Piecewise-Parabolic) streaming quantile estimator (Jain & Chlamtac, 1985).

Why this matters for the "millions of transactions per second" problem:
computing an exact percentile requires storing the whole distribution.
P^2 estimates a single quantile with just 5 running markers — O(1) memory,
O(1) update per transaction — so you can maintain a live "what counts as
suspiciously low/high right now" threshold per account or per shard
without ever storing raw amounts.
"""
from __future__ import annotations


class P2Quantile:
    def __init__(self, p: float):
        assert 0 < p < 1
        self.p = p
        self.n = 0
        self._init_buf: list[float] = []
        self.q = [0.0] * 5
        self.npos = [1, 2, 3, 4, 5]
        self.np_ = [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5]
        self.dn = [0, p / 2, p, (1 + p) / 2, 1]

    def update(self, x: float) -> None:
        if self.n < 5:
            self._init_buf.append(x)
            self.n += 1
            if self.n == 5:
                self._init_buf.sort()
                self.q = list(self._init_buf)
            return

        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            k = 3
            for i in range(1, 5):
                if x < self.q[i]:
                    k = i - 1
                    break

        for i in range(k + 1, 5):
            self.npos[i] += 1
        for i in range(5):
            self.np_[i] += self.dn[i]

        for i in range(1, 4):
            d = self.np_[i] - self.npos[i]
            if (d >= 1 and self.npos[i + 1] - self.npos[i] > 1) or \
               (d <= -1 and self.npos[i - 1] - self.npos[i] < -1):
                d = 1 if d > 0 else -1
                qi = self.q[i] + d / (self.npos[i + 1] - self.npos[i - 1]) * (
                    (self.npos[i] - self.npos[i - 1] + d) * (self.q[i + 1] - self.q[i])
                    / (self.npos[i + 1] - self.npos[i])
                    + (self.npos[i + 1] - self.npos[i] - d) * (self.q[i] - self.q[i - 1])
                    / (self.npos[i] - self.npos[i - 1])
                )
                if self.q[i - 1] < qi < self.q[i + 1]:
                    self.q[i] = qi
                else:
                    step = i + (1 if d > 0 else -1)
                    self.q[i] += d * (self.q[step] - self.q[i]) / (self.npos[step] - self.npos[i])
                self.npos[i] += d
        self.n += 1

    def value(self) -> float:
        if self.n < 5:
            s = sorted(self._init_buf)
            idx = min(int(self.p * len(s)), len(s) - 1) if s else 0
            return s[idx] if s else 0.0
        return self.q[2]


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    data = rng.lognormal(mean=4.0, sigma=1.2, size=20000)  # skewed amount distribution

    low_est, high_est = P2Quantile(0.02), P2Quantile(0.98)
    for x in data:
        low_est.update(float(x))
        high_est.update(float(x))

    print(f"True p2/p98:  {np.percentile(data, 2):.2f} / {np.percentile(data, 98):.2f}")
    print(f"P^2 estimate: {low_est.value():.2f} / {high_est.value():.2f}")
