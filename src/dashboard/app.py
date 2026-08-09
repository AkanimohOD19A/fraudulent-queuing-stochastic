"""
Streamlit dashboard for the fraud-queueing-stochastic project.

Run with:  streamlit run src/dashboard/app.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulate"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "explain"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "trace"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from arrivals import hawkes_stream, mmpp_stream
from queue_model import GatewayQueueSim
from hawkes_fit import flag_windows
from mmpp_fit import fit_mmpp
from harness import compare_models
from cohere_summary import summarize_incident
from langfuse_setup import get_tracer

# Load COHERE_API_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from a
# .env file in the project root, if present. Without this, python only
# sees real environment variables set with `export`, not .env contents.
load_dotenv()

st.set_page_config(page_title="Fraud Queueing & Stochastic Detection", layout="wide")
st.title("Queueing & Stochastic Fraud Detection")
st.caption("Hawkes vs. MMPP burst detection, with a queueing layer and LLM-generated incident summaries.")

with st.sidebar:
    st.header("Simulation settings")
    stream_type = st.selectbox("Generator", ["Hawkes", "MMPP"])
    duration = st.slider("Duration (s)", 60, 600, 300, step=30)
    seed = st.number_input("Random seed", value=11, step=1)
    run_button = st.button("Run simulation", type="primary")

if run_button:
    rng = np.random.default_rng(int(seed))
    tracer = get_tracer()  # real Langfuse tracer if keys are set, else a silent no-op

    with tracer.span("simulate", stream_type=stream_type, duration=duration, seed=int(seed)) as sp:
        if stream_type == "Hawkes":
            attack_start, attack_end = duration * 0.4, duration * 0.5
            events, labels = hawkes_stream(
                mu=0.5, alpha=0.3, beta=1.0, duration=duration, rng=rng,
                attack_windows=[(attack_start, attack_end)], attack_alpha_multiplier=2.5,
            )
        else:
            events, labels = mmpp_stream(
                mu_normal=1.0, mu_attack=8.0, p_normal_to_attack=0.02,
                p_attack_to_normal=0.15, duration=duration, rng=rng,
            )
        sp.update(output={"n_events": len(events), "n_attack_labeled": int(labels.sum())})

    # --- Queueing layer ---
    with tracer.span("queue", n_servers=3, service_rate=1.0) as sp:
        sim = GatewayQueueSim(n_servers=3, service_rate=1.0, rng=rng)
        sim.run(events)
        q_times = [t for t, _ in sim.queue_len_log]
        q_lens = [q for _, q in sim.queue_len_log]
        sp.update(output={
            "mean_wait": float(np.mean(sim.wait_times)) if sim.wait_times else None,
            "max_queue_length": max(q_lens) if q_lens else 0,
        })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction stream")
        st.metric("Total events", len(events))
        st.metric("Labeled attack events", int(labels.sum()))
    with col2:
        st.subheader("Queueing impact")
        st.metric("Mean wait (s)", f"{np.mean(sim.wait_times):.2f}" if sim.wait_times else "n/a")
        st.metric("Max queue length", max(q_lens) if q_lens else 0)

    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(x=q_times, y=q_lens, name="Queue length", line=dict(color="#d62728")))
    fig_queue.update_layout(title="Queue length over time", xaxis_title="time (s)", yaxis_title="queue length")
    st.plotly_chart(fig_queue, use_container_width=True)

    # --- Detection: fit both models ---
    st.subheader("Hawkes vs. MMPP — detection overlay")
    with tracer.span("detect") as sp:
        hawkes_res = flag_windows(events, T=duration)
        mmpp_res = fit_mmpp(events, T=duration)
        sp.update(output={
            "hawkes_flagged_bins": int(hawkes_res["flags"].sum()),
            "mmpp_flagged_bins": int(mmpp_res["flags"].sum()),
        })

    fig_flags = go.Figure()
    fig_flags.add_trace(go.Scatter(
        x=hawkes_res["grid"], y=hawkes_res["flags"].astype(int),
        name="Hawkes flag", line=dict(color="#1f77b4", shape="hv")))
    fig_flags.add_trace(go.Scatter(
        x=mmpp_res["grid"], y=mmpp_res["flags"].astype(int) * 1.05,
        name="MMPP flag", line=dict(color="#2ca02c", shape="hv")))
    if stream_type == "Hawkes":
        fig_flags.add_vrect(x0=attack_start, x1=attack_end, fillcolor="grey", opacity=0.2,
                             annotation_text="ground truth attack", line_width=0)
    fig_flags.update_layout(title="Flagged windows vs. ground truth", xaxis_title="time (s)",
                             yaxis_title="flagged (0/1)")
    st.plotly_chart(fig_flags, use_container_width=True)

    # --- Comparison table ---
    st.subheader("Model comparison")
    with tracer.span("evaluate") as sp:
        comparison_df = compare_models(events, labels, T=duration)
        sp.update(output=comparison_df.to_dict(orient="records"))
    st.dataframe(comparison_df, use_container_width=True)

    # --- Incident explanation ---
    st.subheader("Incident summary (Cohere)")
    with tracer.span("explain") as sp:
        if mmpp_res["flags"].any():
            flagged_idx = np.where(mmpp_res["flags"])[0]
            w_start = mmpp_res["grid"][flagged_idx[0]]
            w_end = mmpp_res["grid"][flagged_idx[-1]]
            rate_mult = mmpp_res["means"][mmpp_res["attack_state"]] / max(
                mmpp_res["means"][1 - mmpp_res["attack_state"]], 1e-6
            )
            summary = summarize_incident(
                model_name="MMPP", window_start=float(w_start), window_end=float(w_end),
                rate_multiple=float(rate_mult), value_delta_pct=-65.0,
            )
            sp.update(output={"summary": summary})
            st.info(summary)
        else:
            sp.update(output={"summary": None, "reason": "no windows flagged"})
            st.write("No windows flagged by MMPP in this run.")

    tracer.flush()  # ensure the trace is sent before the script run ends
else:
    st.write("Set your parameters in the sidebar and click **Run simulation**.")
