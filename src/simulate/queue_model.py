"""
M/M/c queueing model of a payment gateway's authorization capacity.

Takes a pre-generated arrival-time array (from arrivals.py) and replays it
through a SimPy resource pool, recording per-transaction wait time and a
time series of queue length. A burst in arrivals (Hawkes or MMPP attack
regime) shows up here as rising wait times and queue saturation — an
independent signal from the statistical detectors in detect/.
"""
from __future__ import annotations
import numpy as np
import simpy


class GatewayQueueSim:
    def __init__(self, n_servers: int, service_rate: float, rng: np.random.Generator):
        """
        n_servers: number of parallel authorization workers (the 'c' in M/M/c)
        service_rate: mean transactions/sec each server can process (exponential service time)
        """
        self.n_servers = n_servers
        self.service_rate = service_rate
        self.rng = rng
        self.wait_times: list[float] = []
        self.queue_len_log: list[tuple[float, int]] = []  # (time, queue length)

    def _transaction(self, env: simpy.Environment, resource: simpy.Resource, arrival_time: float):
        queue_enter = env.now
        with resource.request() as req:
            yield req
            wait = env.now - queue_enter
            self.wait_times.append(wait)
            service_time = self.rng.exponential(1.0 / self.service_rate)
            yield env.timeout(service_time)

    def run(self, arrival_times: np.ndarray, sample_dt: float = 1.0) -> None:
        env = simpy.Environment()
        resource = simpy.Resource(env, capacity=self.n_servers)

        def arrival_process():
            prev_t = 0.0
            for t in arrival_times:
                yield env.timeout(max(t - prev_t, 0.0))
                prev_t = t
                env.process(self._transaction(env, resource, t))

        def queue_length_sampler():
            while True:
                self.queue_len_log.append((env.now, len(resource.queue)))
                yield env.timeout(sample_dt)

        env.process(arrival_process())
        env.process(queue_length_sampler())
        end_time = float(arrival_times[-1]) + 5.0 if len(arrival_times) else 1.0
        env.run(until=end_time)

    def utilization(self) -> float:
        """rho = lambda / (c * mu), using the empirical arrival rate."""
        if not self.wait_times:
            return 0.0
        implied_lambda = len(self.wait_times) / max(
            (self.queue_len_log[-1][0] if self.queue_len_log else 1.0), 1e-6
        )
        return implied_lambda / (self.n_servers * self.service_rate)


if __name__ == "__main__":
    from arrivals import hawkes_stream

    rng = np.random.default_rng(7)
    events, labels = hawkes_stream(
        mu=0.5, alpha=0.3, beta=1.0, duration=120.0, rng=rng,
        attack_windows=[(40.0, 55.0)], attack_alpha_multiplier=2.5,
    )

    sim = GatewayQueueSim(n_servers=3, service_rate=1.0, rng=rng)
    sim.run(events)

    print(f"Transactions processed: {len(sim.wait_times)}")
    print(f"Mean wait: {np.mean(sim.wait_times):.3f}s | Max wait: {np.max(sim.wait_times):.3f}s")
    print(f"Approx utilization (rho): {sim.utilization():.2f}")
    max_q = max(q for _, q in sim.queue_len_log)
    print(f"Max observed queue length: {max_q}")
