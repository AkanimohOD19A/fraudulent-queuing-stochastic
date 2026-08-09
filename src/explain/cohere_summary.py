"""
Turn a flagged anomaly window's stats into a short plain-language incident
summary using Cohere's command-r. Falls back to a templated summary if no
API key is configured, so the rest of the pipeline (and the dashboard)
still runs end-to-end without Cohere access.

Set COHERE_API_KEY in your environment before using the live path.
"""
from __future__ import annotations
import os
import duckdb

_CACHE_DB = "explanations_cache.duckdb"


def _ensure_cache_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS explanations (
            window_key TEXT PRIMARY KEY,
            summary TEXT
        )
        """
    )


def _window_key(model_name: str, window_start: float, window_end: float) -> str:
    return f"{model_name}:{window_start:.1f}-{window_end:.1f}"


def _fallback_summary(rate_multiple: float, duration: float, value_delta_pct: float) -> str:
    return (
        f"Arrival rate rose to roughly {rate_multiple:.1f}x baseline over "
        f"{duration:.0f} seconds, with mean transaction value "
        f"{'down' if value_delta_pct < 0 else 'up'} {abs(value_delta_pct):.0f}% "
        f"vs. typical — a pattern consistent with a card-testing burst."
    )


def summarize_incident(
    model_name: str,
    window_start: float,
    window_end: float,
    rate_multiple: float,
    value_delta_pct: float,
    use_cache: bool = True,
) -> str:
    """
    rate_multiple: observed arrival rate / baseline rate during the window
    value_delta_pct: % change in mean transaction value during the window
                      vs. baseline (negative = smaller amounts, typical of
                      card-testing attacks probing for valid cards)
    """
    duration = window_end - window_start
    key = _window_key(model_name, window_start, window_end)

    con = duckdb.connect(_CACHE_DB) if use_cache else None
    if con is not None:
        _ensure_cache_table(con)
        cached = con.execute(
            "SELECT summary FROM explanations WHERE window_key = ?", [key]
        ).fetchone()
        if cached:
            con.close()
            return cached[0]

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        summary = _fallback_summary(rate_multiple, duration, value_delta_pct)
    else:
        try:
            import cohere
            client = cohere.Client(api_key)
            prompt = (
                f"In 2-3 sentences, describe this transaction-monitoring incident "
                f"for a fraud analyst. Model that flagged it: {model_name}. "
                f"Arrival rate was {rate_multiple:.1f}x baseline for {duration:.0f} seconds. "
                f"Mean transaction value changed {value_delta_pct:+.0f}% vs. typical. "
                f"Note whether this looks like a card-testing pattern."
            )
            resp = client.chat(model="command-r", message=prompt)
            summary = resp.text.strip()
        except Exception as exc:  # network/quota/etc — degrade gracefully
            summary = _fallback_summary(rate_multiple, duration, value_delta_pct) + \
                f" (Cohere call failed, used fallback: {exc})"

    if con is not None:
        con.execute("INSERT OR REPLACE INTO explanations VALUES (?, ?)", [key, summary])
        con.close()

    return summary


if __name__ == "__main__":
    print(summarize_incident(
        model_name="MMPP", window_start=120.0, window_end=150.0,
        rate_multiple=8.0, value_delta_pct=-70.0,
    ))
