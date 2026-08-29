"""
TwinPulse — Adaptive Digital Twin for Vehicle Assembly Lines
Accenture Innovation Challenge 2026 — Problem Track 4: DigitalTwin.ai

Single-file Streamlit prototype. Synthetic data, explainable statistics only.
No ML training, no external services, no PLC integration.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_STATIONS = 20
N_VEHICLES = 300
SEED = 42
DEGRADED_STATION = 8          # ground-truth injected fault (used only for data-gen)
DOWNSTREAM_DEFECT_STATION = 15
BASELINE_WINDOW = 50           # first N vehicles used as "clean history" reference

st.set_page_config(
    page_title="TwinPulse — Adaptive Digital Twin",
    page_icon="🏭",
    layout="wide",
)

# ----------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ----------------------------------------------------------------------------
@st.cache_data
def generate_data(seed=SEED):
    rng = np.random.default_rng(seed)

    vehicle_ids = np.arange(1, N_VEHICLES + 1)
    station_ids = np.arange(1, N_STATIONS + 1)

    rows = []

    # Per-station baseline "personality" so the line looks realistic, not flat
    station_cycle_base = 55 + rng.normal(0, 3, N_STATIONS)
    station_queue_base = 4 + rng.normal(0, 1, N_STATIONS)
    station_throughput_base = 100 + rng.normal(0, 4, N_STATIONS)
    station_torque_base = 50 + rng.normal(0, 2, N_STATIONS)
    station_temp_base = 68 + rng.normal(0, 2, N_STATIONS)
    station_vibration_base = 0.45 + rng.normal(0, 0.03, N_STATIONS)

    for v in vehicle_ids:
        # progression of this vehicle through time (0 -> 1 across the run)
        progress = (v - 1) / (N_VEHICLES - 1)

        # Station-8 degradation ramps up starting ~1/6th through the run
        ramp_start = 0.15
        degradation = 0.0
        if progress > ramp_start:
            degradation = min(1.0, (progress - ramp_start) / (1 - ramp_start))
            degradation = degradation ** 1.3  # gentle then accelerating

        for s in station_ids:
            idx = s - 1
            is_degraded_station = (s == DEGRADED_STATION)

            cycle_time = station_cycle_base[idx] + rng.normal(0, 1.5)
            queue_length = station_queue_base[idx] + rng.normal(0, 0.6)
            throughput = station_throughput_base[idx] + rng.normal(0, 2.5)
            torque = station_torque_base[idx] + rng.normal(0, 1.2)
            temperature = station_temp_base[idx] + rng.normal(0, 1.0)
            vibration = station_vibration_base[idx] + rng.normal(0, 0.02)

            if is_degraded_station:
                cycle_time += degradation * 14          # up to +~25%
                queue_length += degradation * 6.5        # up to +~40%
                torque += rng.normal(0, 1.0 + degradation * 4.5)  # variability up
                vibration += degradation * 0.55           # up sharply
                temperature += degradation * 5.0
                throughput -= degradation * 18            # station itself slows

            # Downstream stations (after the degraded one) feel reduced throughput
            if s > DEGRADED_STATION:
                distance_decay = max(0, 1 - 0.06 * (s - DEGRADED_STATION))
                throughput -= degradation * 10 * distance_decay
                queue_length += degradation * 1.5 * distance_decay

            rows.append({
                "vehicle_id": v,
                "station_id": s,
                "cycle_time": max(cycle_time, 1),
                "queue_length": max(queue_length, 0),
                "throughput": max(throughput, 0),
                "torque": torque,
                "temperature": temperature,
                "vibration": max(vibration, 0),
                "_degradation": degradation,
            })

    df = pd.DataFrame(rows)

    # ---- Defect generation: concentrated downstream, correlated with the
    # upstream degradation experienced by that SAME vehicle at the fault station
    veh_degradation = df[df.station_id == DEGRADED_STATION].set_index("vehicle_id")["_degradation"]
    df["defect"] = 0
    defect_station_mask = df["station_id"].between(DOWNSTREAM_DEFECT_STATION - 1, N_STATIONS)
    for v in vehicle_ids:
        deg = veh_degradation.get(v, 0)
        base_defect_prob = 0.02
        defect_prob = base_defect_prob + deg * 0.35
        vehicle_defect_rows = df[(df.vehicle_id == v) & defect_station_mask].index
        if len(vehicle_defect_rows) > 0 and rng.random() < defect_prob:
            # defect surfaces at a random downstream inspection point
            chosen = rng.choice(vehicle_defect_rows)
            df.loc[chosen, "defect"] = 1

    return df


@st.cache_data
def apply_sensor_gap(df):
    """Blank out vibration & temperature at the degraded station to simulate a
    real sensor outage, then infer them from available signals with a
    transparent, explainable confidence score."""
    df = df.copy()
    df["vibration_measured"] = df["vibration"]
    df["temperature_measured"] = df["temperature"]

    mask = df["station_id"] == DEGRADED_STATION
    df.loc[mask, "vibration"] = np.nan
    df.loc[mask, "temperature"] = np.nan
    df["vibration_source"] = np.where(mask, "AI Inferred", "Measured")
    df["temperature_source"] = np.where(mask, "AI Inferred", "Measured")

    neighbor_before = DEGRADED_STATION - 1
    neighbor_after = DEGRADED_STATION + 1

    for col in ["vibration", "temperature"]:
        measured_col = f"{col}_measured"
        conf_col = f"{col}_confidence"
        df[conf_col] = np.nan

        nb_before = df[df.station_id == neighbor_before].set_index("vehicle_id")[measured_col]
        nb_after = df[df.station_id == neighbor_after].set_index("vehicle_id")[measured_col]

        # rolling historical average of the degraded station's own values,
        # computed only from vehicles BEFORE the sensor went dark conceptually
        # (here: a trailing rolling mean over vehicle order, using neighbor-informed proxy)
        hist_ref = (nb_before + nb_after) / 2
        rolling_hist = hist_ref.rolling(window=15, min_periods=3, center=False).mean()
        rolling_hist = rolling_hist.fillna(hist_ref.mean())

        # Reference point representing "quiet" plant behavior, used only to
        # gauge how far the current operating point has drifted (drift makes
        # a neighbor-based proxy less trustworthy — data availability alone
        # isn't enough once the underlying process has clearly shifted).
        quiet_reference = hist_ref.iloc[:BASELINE_WINDOW].mean()
        max_drift = (hist_ref - quiet_reference).abs().max()

        for v in df.loc[mask, "vehicle_id"].unique():
            sources = np.array([
                nb_before.get(v, np.nan),
                nb_after.get(v, np.nan),
                rolling_hist.get(v, np.nan),
            ])
            sources = sources[~np.isnan(sources)]
            if len(sources) == 0:
                continue
            estimate = float(np.mean(sources))

            # --- transparent confidence rule based on data availability ---
            n_sources = len(sources)
            if n_sources > 1 and np.mean(sources) != 0:
                disagreement = np.std(sources) / (abs(np.mean(sources)) + 1e-6)
                agreement = max(0.0, 1 - disagreement)
            else:
                agreement = 0.5  # only one weak source available

            drift_now = abs(hist_ref.get(v, quiet_reference) - quiet_reference)
            drift_penalty = (drift_now / max_drift) if max_drift > 0 else 0.0

            confidence = 55 + (n_sources / 3) * 20 + agreement * 20 - drift_penalty * 30
            confidence = float(np.clip(confidence, 40, 95))

            row_idx = df[(df.vehicle_id == v) & (df.station_id == DEGRADED_STATION)].index
            df.loc[row_idx, col] = estimate
            df.loc[row_idx, conf_col] = confidence

    return df


# ----------------------------------------------------------------------------
# 2. ANALYTICS (simple, explainable statistics — no black-box ML)
# ----------------------------------------------------------------------------
def normalize(series, low=None, high=None):
    low = series.min() if low is None else low
    high = series.max() if high is None else high
    if high - low == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - low) / (high - low)).clip(0, 1)


@st.cache_data
def compute_station_metrics(df):
    """One row per station: baseline-relative deviations + composite risk scores."""
    baseline_df = df[df.vehicle_id <= BASELINE_WINDOW]
    recent_df = df[df.vehicle_id > N_VEHICLES - 60]  # "current" operating window

    baseline = baseline_df.groupby("station_id").agg(
        base_cycle=("cycle_time", "mean"),
        base_queue=("queue_length", "mean"),
        base_throughput=("throughput", "mean"),
        base_torque_std=("torque", "std"),
        base_temp=("temperature_measured", "mean"),
        base_vibration=("vibration_measured", "mean"),
    )

    recent = recent_df.groupby("station_id").agg(
        cycle_time=("cycle_time", "mean"),
        queue_length=("queue_length", "mean"),
        throughput=("throughput", "mean"),
        torque_std=("torque", "std"),
        torque_mean=("torque", "mean"),
        temperature=("temperature", "mean"),
        vibration=("vibration", "mean"),
        defect_rate=("defect", "mean"),
    )

    m = baseline.join(recent)

    # --- relative deviations vs each station's own clean baseline ---
    m["cycle_dev_pct"] = (m["cycle_time"] - m["base_cycle"]) / m["base_cycle"] * 100
    m["queue_dev_pct"] = (m["queue_length"] - m["base_queue"]) / m["base_queue"] * 100
    m["throughput_dev_pct"] = (m["throughput"] - m["base_throughput"]) / m["base_throughput"] * 100
    m["torque_var_dev_pct"] = (m["torque_std"] - m["base_torque_std"]) / m["base_torque_std"].replace(0, np.nan) * 100
    m["temp_dev_pct"] = (m["temperature"] - m["base_temp"]) / m["base_temp"] * 100
    m["vibration_dev_pct"] = (m["vibration"] - m["base_vibration"]) / m["base_vibration"] * 100
    m = m.fillna(0)

    # --- Anomaly score: average of |z-like deviations| across signals, 0-100 ---
    dev_cols = ["cycle_dev_pct", "queue_dev_pct", "torque_var_dev_pct", "vibration_dev_pct", "temp_dev_pct"]
    m["anomaly_raw"] = m[dev_cols].abs().mean(axis=1)
    m["anomaly_score"] = (normalize(m["anomaly_raw"]) * 100).round(1)

    # --- Bottleneck risk: cycle-time up + queue up + throughput down ---
    bn_raw = (
        normalize(m["cycle_dev_pct"].clip(lower=0)) * 0.4
        + normalize(m["queue_dev_pct"].clip(lower=0)) * 0.35
        + normalize((-m["throughput_dev_pct"]).clip(lower=0)) * 0.25
    )
    m["bottleneck_risk"] = (bn_raw * 100).round(1)

    # --- Defect risk: torque variability + temp + vibration + cycle-time anomalies ---
    dr_raw = (
        normalize(m["torque_var_dev_pct"].clip(lower=0)) * 0.3
        + normalize(m["temp_dev_pct"].abs()) * 0.2
        + normalize(m["vibration_dev_pct"].clip(lower=0)) * 0.3
        + normalize(m["cycle_dev_pct"].clip(lower=0)) * 0.2
    )
    m["defect_risk"] = (dr_raw * 100).round(1)

    # --- Status classification ---
    def classify(row):
        if row["bottleneck_risk"] >= 55 or row["defect_risk"] >= 55:
            return "🔴 Critical"
        elif row["bottleneck_risk"] >= 30 or row["defect_risk"] >= 30:
            return "🟡 Warning"
        return "🟢 Normal"

    m["status"] = m.apply(classify, axis=1)
    m = m.reset_index()
    return m


@st.cache_data
def root_cause_analysis(station_metrics, affected_station=DOWNSTREAM_DEFECT_STATION):
    """Rank upstream stations as likely contributors using anomaly score,
    weighted by proximity to the affected downstream station."""
    upstream = station_metrics[station_metrics.station_id < affected_station].copy()
    upstream["distance"] = affected_station - upstream["station_id"]
    # Score = anomaly strength, gently discounted by distance (closer stations get
    # a small boost, but a strongly anomalous far station can still dominate)
    upstream["contribution_score"] = upstream["anomaly_score"] / (1 + 0.03 * upstream["distance"])
    upstream = upstream.sort_values("contribution_score", ascending=False)

    top = upstream.iloc[0]
    total_score = upstream["contribution_score"].sum()
    confidence = float(np.clip((top["contribution_score"] / total_score) * 100 * 3, 40, 96)) if total_score > 0 else 50

    return upstream, top, round(confidence, 1)


# ----------------------------------------------------------------------------
# 3. WHAT-IF SIMULATION
# ----------------------------------------------------------------------------
def simulate_intervention(station_metrics, option, station_id=DEGRADED_STATION):
    row = station_metrics[station_metrics.station_id == station_id].iloc[0].copy()

    sim_cycle_dev = row["cycle_dev_pct"]
    sim_queue_dev = row["queue_dev_pct"]
    sim_throughput_dev = row["throughput_dev_pct"]
    sim_torque_var_dev = row["torque_var_dev_pct"]
    sim_vibration_dev = row["vibration_dev_pct"]
    sim_temp_dev = row["temp_dev_pct"]

    if option == "A":  # reduce cycle time by 10%
        sim_cycle_dev *= 0.90
        sim_queue_dev *= 0.82          # shorter cycle relieves queue buildup
        sim_throughput_dev = sim_throughput_dev + abs(sim_throughput_dev) * 0.18
        sim_torque_var_dev *= 0.88
        sim_vibration_dev *= 0.90
    elif option == "B":  # increase capacity by 10%
        sim_queue_dev *= 0.70          # extra capacity absorbs queue directly
        sim_throughput_dev = sim_throughput_dev + abs(sim_throughput_dev) * 0.28
        sim_cycle_dev *= 0.96
        sim_torque_var_dev *= 0.95
        sim_vibration_dev *= 0.97
    # option "C" = no change, leave as-is

    bn_raw = (
        normalize_single(sim_cycle_dev, station_metrics["cycle_dev_pct"]) * 0.4
        + normalize_single(sim_queue_dev, station_metrics["queue_dev_pct"]) * 0.35
        + normalize_single(-sim_throughput_dev, -station_metrics["throughput_dev_pct"]) * 0.25
    )
    dr_raw = (
        normalize_single(sim_torque_var_dev, station_metrics["torque_var_dev_pct"]) * 0.3
        + normalize_single(abs(sim_temp_dev), station_metrics["temp_dev_pct"].abs()) * 0.2
        + normalize_single(sim_vibration_dev, station_metrics["vibration_dev_pct"]) * 0.3
        + normalize_single(sim_cycle_dev, station_metrics["cycle_dev_pct"]) * 0.2
    )

    sim_bottleneck = float(np.clip(bn_raw * 100, 0, 100))
    sim_defect = float(np.clip(dr_raw * 100, 0, 100))
    sim_throughput = row["throughput"] * (1 + (0 if option == "C" else 0.06 if option == "A" else 0.09))

    return {
        "bottleneck_risk": round(sim_bottleneck, 1),
        "defect_risk": round(sim_defect, 1),
        "throughput": round(sim_throughput, 1),
    }


def normalize_single(value, reference_series, low=None, high=None):
    low = reference_series.clip(lower=0).min() if low is None else low
    high = reference_series.clip(lower=0).max() if high is None else high
    v = max(value, 0)
    if high - low == 0:
        return 0.0
    return float(np.clip((v - low) / (high - low), 0, 1))


# ----------------------------------------------------------------------------
# LOAD / PREP DATA
# ----------------------------------------------------------------------------
raw_df = generate_data()
df = apply_sensor_gap(raw_df)
station_metrics = compute_station_metrics(df)
upstream_ranked, top_contributor, rc_confidence = root_cause_analysis(station_metrics)

overall_throughput = station_metrics["throughput"].mean()
max_bottleneck_row = station_metrics.loc[station_metrics["bottleneck_risk"].idxmax()]
max_defect_row = station_metrics.loc[station_metrics["defect_risk"].idxmax()]
active_anomalies = int((station_metrics["status"] != "🟢 Normal").sum())

# ----------------------------------------------------------------------------
# UI — HEADER
# ----------------------------------------------------------------------------
st.markdown("## 🏭 TWINPULSE")
st.markdown("##### *Adaptive Digital Twin for Vehicle Assembly Lines*")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Throughput (avg)", f"{overall_throughput:.1f} units/hr")
c2.metric("Highest Bottleneck Risk", f"{max_bottleneck_row['bottleneck_risk']:.0f}%", f"Station {int(max_bottleneck_row['station_id'])}")
c3.metric("Highest Defect Risk", f"{max_defect_row['defect_risk']:.0f}%", f"Station {int(max_defect_row['station_id'])}")
c4.metric("Active Anomalies", f"{active_anomalies} stations")

st.warning(f"⚠️ Early warning: Station {int(max_bottleneck_row['station_id'])} shows abnormal production behavior — bottleneck risk {max_bottleneck_row['bottleneck_risk']:.0f}%, trending upward.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🏭 Digital Twin", "🔮 Predictions", "🔍 Root Cause", "🧪 What-If"])

# ----------------------------------------------------------------------------
# TAB 1 — DIGITAL TWIN
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Live Production Flow")

    flow_html = ""
    for _, r in station_metrics.sort_values("station_id").iterrows():
        icon = r["status"].split()[0]
        flow_html += f"**S{int(r['station_id']):02d}** {icon}"
        if r["station_id"] != N_STATIONS:
            flow_html += " → "
    st.markdown(f"<div style='font-size:20px; line-height:2.2;'>{flow_html}</div>", unsafe_allow_html=True)

    st.caption("🟢 Normal · 🟡 Warning · 🔴 Critical — status derived from bottleneck & defect risk scores")

    st.divider()

    selected_station = st.selectbox(
        "Select a station to inspect",
        options=station_metrics["station_id"].tolist(),
        index=int(DEGRADED_STATION - 1),
        format_func=lambda x: f"Station {x:02d}",
    )

    srow = station_metrics[station_metrics.station_id == selected_station].iloc[0]
    coverage_measured = df[(df.station_id == selected_station)]["vibration_source"].eq("Measured").mean() * 100

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Cycle Time", f"{srow['cycle_time']:.1f} s", f"{srow['cycle_dev_pct']:+.1f}% vs baseline")
        st.metric("Queue Length", f"{srow['queue_length']:.1f}", f"{srow['queue_dev_pct']:+.1f}% vs baseline")
    with colB:
        st.metric("Throughput", f"{srow['throughput']:.1f} units/hr", f"{srow['throughput_dev_pct']:+.1f}% vs baseline")
        st.metric("Sensor Coverage", f"{coverage_measured:.0f}% measured", f"{100 - coverage_measured:.0f}% AI inferred")
    with colC:
        st.metric("Anomaly Score", f"{srow['anomaly_score']:.0f}/100")
        st.metric("Bottleneck Risk", f"{srow['bottleneck_risk']:.0f}%")

    st.markdown(f"**Status: {srow['status']}**")

    if selected_station == DEGRADED_STATION:
        st.info("🔬 This station has partial sensor coverage. Vibration & temperature are being estimated — see the **Root Cause** tab for measured vs. AI-inferred values.")

    # Trend chart across vehicle order for the selected station
    st_df = df[df.station_id == selected_station].sort_values("vehicle_id")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st_df.vehicle_id, y=st_df.cycle_time, name="Cycle Time", line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=st_df.vehicle_id, y=st_df.queue_length * 10, name="Queue Length (x10)", line=dict(color="#f39c12")))
    fig.add_trace(go.Scatter(x=st_df.vehicle_id, y=st_df.throughput, name="Throughput", line=dict(color="#2ecc71")))
    fig.update_layout(
        title=f"Station {selected_station:02d} — Trend Across Production Run",
        xaxis_title="Vehicle #", yaxis_title="Value",
        height=380, legend=dict(orientation="h", y=1.12), margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — PREDICTIONS
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Predictive Signals")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Bottleneck Risk", f"{max_bottleneck_row['bottleneck_risk']:.0f}%", f"Station {int(max_bottleneck_row['station_id']):02d}")
    with p2:
        st.metric("Defect Risk", f"{max_defect_row['defect_risk']:.0f}%", "Downstream batch")
    with p3:
        st.metric("Anomalies Detected", f"{active_anomalies}", "stations flagged")

    st.divider()
    st.markdown("#### Why this prediction was made")

    fs = station_metrics[station_metrics.station_id == max_bottleneck_row["station_id"]].iloc[0]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Cycle Time", f"{fs['cycle_dev_pct']:+.1f}%", "vs baseline")
    r2.metric("Queue Length", f"{fs['queue_dev_pct']:+.1f}%", "vs baseline")
    r3.metric("Throughput", f"{fs['throughput_dev_pct']:+.1f}%", "vs baseline")
    r4.metric("Torque Variability", f"{fs['torque_var_dev_pct']:+.1f}%", "vs baseline")

    st.caption(
        f"Station {int(fs['station_id']):02d} shows cycle time up {fs['cycle_dev_pct']:.0f}%, "
        f"queue up {fs['queue_dev_pct']:.0f}%, and throughput down {abs(fs['throughput_dev_pct']):.0f}% "
        f"relative to its own historical baseline — the combination that drives the elevated bottleneck score."
    )

    st.divider()
    st.markdown("#### Risk Across All Stations")
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=station_metrics.station_id, y=station_metrics.bottleneck_risk, name="Bottleneck Risk", marker_color="#e67e22"))
    bar_fig.add_trace(go.Bar(x=station_metrics.station_id, y=station_metrics.defect_risk, name="Defect Risk", marker_color="#c0392b"))
    bar_fig.update_layout(barmode="group", xaxis_title="Station", yaxis_title="Risk (%)", height=380, legend=dict(orientation="h", y=1.12), margin=dict(t=60))
    st.plotly_chart(bar_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — ROOT CAUSE
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Root-Cause Trace")

    chain_stations = sorted({int(top_contributor["station_id"]), 10, 12, DOWNSTREAM_DEFECT_STATION})
    chain_str = "  →  ".join([f"**S{s:02d}**" + ("  🎯" if s == int(top_contributor['station_id']) else "") for s in chain_stations])
    st.markdown(f"<div style='font-size:19px'>{chain_str}</div>", unsafe_allow_html=True)
    st.caption(f"Tracing backward from the downstream defect concentration near Station {DOWNSTREAM_DEFECT_STATION:02d}.")

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Likely Contributor")
        st.markdown(f"# Station {int(top_contributor['station_id']):02d}")
        st.markdown(f"**Confidence: {rc_confidence:.0f}%**")
        st.caption('Labeled "likely contributor" — this is a statistical inference, not a proven root cause.')

    with col2:
        st.markdown("### Supporting Signals")
        st.markdown(f"- Cycle time ↑ **{top_contributor['cycle_dev_pct']:+.1f}%**")
        st.markdown(f"- Queue growth ↑ **{top_contributor['queue_dev_pct']:+.1f}%**")
        st.markdown(f"- Torque variability ↑ **{top_contributor['torque_var_dev_pct']:+.1f}%**")
        st.markdown(f"- Downstream throughput ↓ **{top_contributor['throughput_dev_pct']:+.1f}%**")
        st.markdown(f"- Contribution score: **{top_contributor['contribution_score']:.1f}**")

    st.divider()
    st.markdown("### Upstream Station Ranking")
    rank_display = upstream_ranked[["station_id", "anomaly_score", "distance", "contribution_score", "status"]].copy()
    rank_display.columns = ["Station", "Anomaly Score", "Distance from Defect", "Contribution Score", "Status"]
    st.dataframe(rank_display.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"### Measured vs. AI-Inferred — Station {int(top_contributor['station_id']):02d}")
    focus_df = df[df.station_id == top_contributor["station_id"]].sort_values("vehicle_id")

    m1, m2 = st.columns(2)
    with m1:
        avg_conf_vib = focus_df["vibration_confidence"].mean()
        st.markdown(f"**Vibration:** 🤖 AI Inferred")
        st.markdown(f"**Confidence: {avg_conf_vib:.0f}%**")
        vib_fig = go.Figure()
        vib_fig.add_trace(go.Scatter(x=focus_df.vehicle_id, y=focus_df.vibration, name="Inferred Vibration", line=dict(color="#9b59b6", dash="dot")))
        vib_fig.update_layout(height=280, margin=dict(t=20), yaxis_title="Vibration")
        st.plotly_chart(vib_fig, use_container_width=True)
    with m2:
        avg_conf_temp = focus_df["temperature_confidence"].mean()
        st.markdown(f"**Temperature:** 🤖 AI Inferred")
        st.markdown(f"**Confidence: {avg_conf_temp:.0f}%**")
        temp_fig = go.Figure()
        temp_fig.add_trace(go.Scatter(x=focus_df.vehicle_id, y=focus_df.temperature, name="Inferred Temperature", line=dict(color="#e67e22", dash="dot")))
        temp_fig.update_layout(height=280, margin=dict(t=20), yaxis_title="Temperature (°C)")
        st.plotly_chart(temp_fig, use_container_width=True)

    st.caption("Dotted lines indicate AI-inferred values (sensor gap at this station). Confidence is computed from the number of available reference signals (neighboring stations + rolling history) and how closely those sources agree — not from comparison against unavailable ground truth.")

# ----------------------------------------------------------------------------
# TAB 4 — WHAT-IF SIMULATION
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("What-If Simulation")
    st.caption(f"Testing an intervention at the likely-contributor station: **Station {int(top_contributor['station_id']):02d}**. No connection to real machinery or PLCs — statistical projection only.")

    option_label = st.radio(
        "Choose an intervention to simulate",
        options=["A", "B", "C"],
        format_func=lambda x: {
            "A": "Option A — Reduce Station cycle time by 10%",
            "B": "Option B — Increase Station capacity by 10%",
            "C": "Option C — No intervention",
        }[x],
        horizontal=False,
    )

    current = station_metrics[station_metrics.station_id == top_contributor["station_id"]].iloc[0]
    sim = simulate_intervention(station_metrics, option_label, station_id=top_contributor["station_id"])

    st.divider()
    comp_df = pd.DataFrame({
        "Metric": ["Bottleneck Risk (%)", "Throughput (units/hr)", "Defect Risk (%)"],
        "Current": [f"{current['bottleneck_risk']:.1f}", f"{current['throughput']:.1f}", f"{current['defect_risk']:.1f}"],
        "Simulated": [f"{sim['bottleneck_risk']:.1f}", f"{sim['throughput']:.1f}", f"{sim['defect_risk']:.1f}"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    colX, colY, colZ = st.columns(3)
    colX.metric("Bottleneck Risk", f"{sim['bottleneck_risk']:.1f}%", f"{sim['bottleneck_risk'] - current['bottleneck_risk']:+.1f} pts")
    colY.metric("Throughput", f"{sim['throughput']:.1f} units/hr", f"{sim['throughput'] - current['throughput']:+.1f}")
    colZ.metric("Defect Risk", f"{sim['defect_risk']:.1f}%", f"{sim['defect_risk'] - current['defect_risk']:+.1f} pts", delta_color="inverse")

    # Determine overall best option among all three for the recommendation
    results = {}
    for opt in ["A", "B", "C"]:
        r = simulate_intervention(station_metrics, opt, station_id=top_contributor["station_id"])
        # lower is better for both risks, higher is better for throughput
        score = (100 - r["bottleneck_risk"]) + (100 - r["defect_risk"]) + r["throughput"] * 0.5
        results[opt] = (r, score)

    best_option = max(results, key=lambda k: results[k][1])
    best_label = {
        "A": "Option A — Reduce cycle time by 10%",
        "B": "Option B — Increase capacity by 10%",
        "C": "Option C — No intervention",
    }[best_option]

    st.divider()
    st.success(f"✅ **Recommended Intervention: {best_label}**")
    st.caption("Recommendation is based on the combined projected improvement across bottleneck risk, defect risk, and throughput.")

    st.divider()
    st.markdown("#### Current vs. Simulated — All Options")
    compare_all = pd.DataFrame({
        "Option": ["A — Reduce Cycle Time", "B — Increase Capacity", "C — No Intervention"],
        "Bottleneck Risk (%)": [results["A"][0]["bottleneck_risk"], results["B"][0]["bottleneck_risk"], results["C"][0]["bottleneck_risk"]],
        "Throughput (units/hr)": [results["A"][0]["throughput"], results["B"][0]["throughput"], results["C"][0]["throughput"]],
        "Defect Risk (%)": [results["A"][0]["defect_risk"], results["B"][0]["defect_risk"], results["C"][0]["defect_risk"]],
    })
    st.dataframe(compare_all, use_container_width=True, hide_index=True)

st.divider()
st.caption("TwinPulse — synthetic demonstration data only. Not connected to real machinery, PLCs, or plant systems. All risk scores are derived from explainable statistical rules, not black-box ML.")
