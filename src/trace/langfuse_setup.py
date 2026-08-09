"""
Wraps the full pipeline (simulate -> fit -> detect -> evaluate -> explain)
in Langfuse traces, with each phase as a nested span. Degrades to a no-op
tracer if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set, so the
pipeline still runs without a Langfuse account.

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally
LANGFUSE_HOST in your environment for the live path.
"""
from __future__ import annotations
import os
from contextlib import contextmanager


class _NoOpSpan:
    def update(self, **kwargs): pass
    def end(self, **kwargs): pass


class _NoOpTracer:
    """Used when Langfuse isn't configured — keeps call sites identical."""
    @contextmanager
    def span(self, name: str, **kwargs):
        yield _NoOpSpan()

    def flush(self): pass


class LangfuseTracer:
    def __init__(self):
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        self._enabled = bool(public_key and secret_key)
        self._client = None
        self._trace = None
        if self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                self._trace = self._client.trace(name="fraud-queueing-stochastic-run")
            except Exception as exc:
                print(f"[langfuse_setup] Falling back to no-op tracer: {exc}")
                self._enabled = False

    @contextmanager
    def span(self, name: str, **metadata):
        if not self._enabled:
            yield _NoOpSpan()
            return
        span = self._trace.span(name=name, metadata=metadata)
        try:
            yield span
        finally:
            span.end()

    def flush(self):
        if self._enabled and self._client is not None:
            self._client.flush()


def get_tracer() -> "LangfuseTracer | _NoOpTracer":
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if public_key and secret_key:
        return LangfuseTracer()
    return _NoOpTracer()


if __name__ == "__main__":
    tracer = get_tracer()
    with tracer.span("simulate", detail="hawkes stream, T=300s") as sp:
        sp.update(output={"n_events": 90})
    with tracer.span("fit_and_detect", model="MMPP") as sp:
        sp.update(output={"flagged_bins": 28})
    tracer.flush()
    print("Trace run complete (no-op tracer unless LANGFUSE_* env vars are set).")
