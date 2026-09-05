
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="TwinPulse V3", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:2rem;}
[data-testid="stMetric"] {
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.08);
    padding:13px; border-radius:14px;
}
.station-card {
    border:1px solid rgba(255,255,255,.10);
    border-radius:13px; padding:11px;
    background:rgba(255,255,255,.025);
    min-height:92px;
}
.alert-box {
    padding:16px; border-radius:14px;
    border-left:5px solid #ff5a5f;
    background:rgba(255,90,95,.12);
}
.small-note {opacity:.72; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def make_data():
    rng = np.random.default_rng(42)
    stations = [f"S{i:02d}" for i in range(1,31)]
    rows = []
    for vehicle in range(1,401):
        for i, s in enumerate(stations):
            cycle = max(35, 60 + rng.normal(0, 2.6))
            queue = max(0, int(rng.normal(3, 1.2)))
            torque = 100 + rng.normal(0, 3.0)
            quality_risk = max(0, rng.normal(4, 1.2))

            # Gradual S08 degradation: deliberately synthetic, not sudden
            if s == "S08" and vehicle > 220:
                d = (vehicle - 220) / 180
                cycle += 18*d
                queue += int(8*d)
                torque += 10*d
                quality_risk += 20*d

            # Downstream propagation
            if 8 < i+1 <= 14 and vehicle > 270:
                d = (vehicle - 270) / 130
                cycle += 4*d
                queue += int(3*d)
                quality_risk += 5*d

            rows.append([vehicle, s, cycle, queue, torque, quality_risk])

    return pd.DataFrame(rows, columns=[
        "vehicle","station","cycle_time","queue_length","torque","quality_risk"
    ])

df = make_data()
stations = [f"S{i:02d}" for i in range(1,31)]

# Deliberately uneven sensor coverage across the line
coverage = {}
for i, s in enumerate(stations, 1):
    if s == "S08":
        coverage[s] = "PARTIAL"
    elif i % 7 == 0:
        coverage[s] = "MANUAL"
    elif i % 4 == 0:
        coverage[s] = "PARTIAL"
    else:
        coverage[s] = "HIGH"

st.sidebar.header("Twin Controls")
scenario = st.sidebar.radio(
    "Operating scenario",
    ["Normal line", "S08 degradation scenario"],
    index=1
)
default_progress = 400 if scenario == "S08 degradation scenario" else 220
progress = st.sidebar.slider("Production progress", 1, 400, default_progress)
if scenario == "Normal line":
    progress = min(progress, 220)

view = df[df.vehicle <= progress]
latest = view.sort_values("vehicle").groupby("station").tail(1).set_index("station")

baseline = df[df.vehicle <= 200].groupby("station")[
    ["cycle_time","queue_length","torque","quality_risk"]
].mean()

relative = (latest[["cycle_time","queue_length","torque","quality_risk"]] -
            baseline.reindex(latest.index)) / baseline.reindex(latest.index)

risk = 100 * (
    0.38*np.clip(relative["cycle_time"], 0, 2)/2 +
    0.30*np.clip(relative["queue_length"], 0, 2)/2 +
    0.18*np.clip(relative["torque"], 0, 2)/2 +
    0.14*np.clip(relative["quality_risk"], 0, 2)/2
)

latest["risk"] = risk.clip(0,100).fillna(0)
latest["status"] = np.where(
    latest.risk >= 55, "Critical",
    np.where(latest.risk >= 30, "Watch", "Normal")
)
latest["coverage"] = [coverage[s] for s in latest.index]
priority = latest.risk.idxmax()
line_health = max(0, 100 - latest.risk.mean()*1.8)

st.title("⚙️ TwinPulse")
st.caption("Adaptive digital twin for manufacturing lines • V3 MVP • Synthetic, reproducible production data")

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Active stations", "30")
m2.metric("Vehicles simulated", f"{progress}")
m3.metric("Line health", f"{line_health:.1f}/100")
m4.metric("Highest bottleneck risk", f"{latest.risk.max():.0f}%")
m5.metric("Priority station", priority)

tabs = st.tabs([
    "🏭 Live Line",
    "🗺️ Twin Map",
    "🔍 Diagnostics",
    "🧠 Prediction Cockpit",
    "🎛️ What-If Simulator",
    "📈 Business Impact"
])

with tabs[0]:
    st.subheader("Production Line Digital Twin")
    st.caption("A 30-station representation of current production state.")

    cols = st.columns(6)
    for i,s in enumerate(stations):
        r = latest.loc[s]
        dot = "🔴" if r.status == "Critical" else ("🟡" if r.status == "Watch" else "🟢")
        with cols[i % 6]:
            st.markdown(
                f'<div class="station-card"><b>{dot} {s}</b><br>'
                f'<span class="small-note">{r.status.upper()} • {coverage[s]} COVERAGE</span><br>'
                f'Risk <b>{r.risk:.0f}%</b> • Queue {int(r.queue_length)}</div>',
                unsafe_allow_html=True
            )

    st.divider()
    c1,c2 = st.columns([1.1,1])
    with c1:
        plot = latest.reset_index()
        fig = px.bar(plot, x="station", y="risk", color="status",
                     category_orders={"status":["Normal","Watch","Critical"]},
                     hover_data=["cycle_time","queue_length","torque","quality_risk"],
                     height=360)
        fig.update_layout(yaxis_title="Risk score (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        summary = latest.reset_index()[[
            "station","status","coverage","cycle_time","queue_length","risk"
        ]].copy()
        summary.columns = ["Station","State","Coverage","Cycle time (s)","Queue","Risk (%)"]
        st.dataframe(summary, use_container_width=True, hide_index=True, height=360)

with tabs[1]:
    st.subheader("Connected Production Twin")
    st.caption("TwinPulse interprets each station in the context of its upstream and downstream relationships.")

    positions = {s:(i, 0) for i,s in enumerate(stations)}
    fig = go.Figure()

    # Relationship edges
    for i in range(len(stations)-1):
        x0,y0 = positions[stations[i]]
        x1,y1 = positions[stations[i+1]]
        fig.add_trace(go.Scatter(
            x=[x0,x1], y=[y0,y1], mode="lines",
            line=dict(width=3, color="rgba(150,150,150,.35)"),
            hoverinfo="skip", showlegend=False
        ))

    color_map = {"Normal":"#2ca02c", "Watch":"#f2a900", "Critical":"#d62728"}
    for s in stations:
        x,y = positions[s]
        r = latest.loc[s]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            text=[s], textposition="top center",
            marker=dict(size=26, color=color_map[r.status]),
            hovertemplate=(
                f"<b>{s}</b><br>Risk: {r.risk:.0f}%<br>"
                f"Coverage: {coverage[s]}<br>Queue: {int(r.queue_length)}<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        height=300, margin=dict(l=20,r=20,t=20,b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensor Coverage Architecture")
    cov_df = pd.DataFrame({
        "Station": stations,
        "Coverage": [coverage[s] for s in stations]
    })
    c1,c2 = st.columns([1,1])
    with c1:
        counts = cov_df["Coverage"].value_counts().reset_index()
        counts.columns = ["Coverage","Stations"]
        st.plotly_chart(
            px.bar(counts, x="Coverage", y="Stations", color="Coverage", height=330),
            use_container_width=True
        )
    with c2:
        st.markdown("### Why this matters")
        st.write(
            "**TwinPulse does not assume every station has perfect instrumentation.** "
            "High-coverage stations contribute richer direct signals, partial-coverage stations "
            "combine measured and inferred states, and manual/limited stations can still be interpreted "
            "through production relationships."
        )
        st.info(
            "S08 is the MVP's challenge case: temperature and vibration are unavailable, while "
            "cycle time, queue length and torque remain available."
        )

with tabs[2]:
    st.subheader("S08 Live Diagnostics")
    s08 = view[view.station == "S08"]
    c1,c2 = st.columns([1.15,.85])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s08.vehicle, y=s08.cycle_time,
            name="Cycle time (s)", line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=s08.vehicle, y=s08.queue_length,
            name="Queue length", yaxis="y2", line=dict(width=2)
        ))
        fig.update_layout(
            height=350, xaxis_title="Vehicle sequence",
            yaxis=dict(title="Cycle time"),
            yaxis2=dict(title="Queue", overlaying="y", side="right"),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        r = latest.loc["S08"]
        if scenario == "S08 degradation scenario" and progress > 220:
            st.markdown(
                f'<div class="alert-box"><b>🚨 Emerging bottleneck detected at S08</b><br>'
                f'Risk score: <b>{r.risk:.0f}%</b><br>'
                f'Inference confidence: <b>82%</b></div>',
                unsafe_allow_html=True
            )
            st.write("**Observed behavioural drift**")
            st.write("• Cycle time is increasing")
            st.write("• Queue growth is accelerating")
            st.write("• Torque has deviated from baseline")
            st.write("• Quality-risk proxy is rising")
            st.write("• Temperature and vibration are unavailable")
        else:
            st.success("No elevated S08 risk in the current production window.")

    st.subheader("Measured vs AI-Inferred State")
    r = latest.loc["S08"]
    inferred_temp = 65 + 0.15*max(r.cycle_time-60,0)
    inferred_vib = 1.20 + 0.02*max(r.torque-100,0)

    table = pd.DataFrame({
        "Signal":["Cycle time","Queue length","Torque","Temperature","Vibration"],
        "Source":["Measured","Measured","Measured","AI-inferred","AI-inferred"],
        "Current state":[
            f"{r.cycle_time:.1f} s", f"{int(r.queue_length)}", f"{r.torque:.1f}",
            f"{inferred_temp:.1f} °C", f"{inferred_vib:.2f}"
        ],
        "Confidence":["100%","100%","100%","82%","82%"]
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("TwinPulse Prediction Cockpit")
    r = latest.loc["S08"]

    a,b,c,d = st.columns(4)
    a.metric("S08 bottleneck risk", f"{r.risk:.0f}%")
    b.metric("Inference confidence", "82%")
    c.metric("Quality-risk proxy", f"{r.quality_risk:.0f}%")
    d.metric("Decision mode", "Recommend")

    st.markdown("### Explainable contributing signals")
    e1,e2,e3,e4 = st.columns(4)
    e1.metric("Cycle-time deviation", f"{max(relative.loc['S08','cycle_time']*100,0):.1f}%")
    e2.metric("Queue deviation", f"{max(relative.loc['S08','queue_length']*100,0):.1f}%")
    e3.metric("Torque deviation", f"{max(relative.loc['S08','torque']*100,0):.1f}%")
    e4.metric("Quality-risk deviation", f"{max(relative.loc['S08','quality_risk']*100,0):.1f}%")

    st.markdown("### Origin and downstream exposure")
    trace = pd.DataFrame({
        "Twin stage":["Likely origin","Immediate exposure","Propagation window","Recommended action"],
        "Finding":["S08 behavioural drift","S09 → S11","S12 → S14",
                   "Inspect/service S08 before further queue propagation"]
    })
    st.dataframe(trace, use_container_width=True, hide_index=True)

    st.info(
        "AI supports plant decisions rather than replacing accountability: low-risk actions may be automated "
        "in future validated deployments, medium-risk actions are recommended for review, and high-consequence "
        "actions require human approval."
    )

with tabs[4]:
    st.subheader("What-If Intervention Simulator")
    st.caption(
        "Interactive prototype scenario: this is a decision-support simulation using the synthetic MVP model, "
        "not a validated prediction of a live factory."
    )

    intervention = st.selectbox(
        "Choose an intervention",
        [
            "Inspect and service S08 now",
            "Increase S08 effective capacity",
            "Reduce upstream release rate",
            "No intervention"
        ]
    )

    base_risk = float(latest.loc["S08","risk"])
    base_queue = float(latest.loc["S08","queue_length"])
    base_exposure = 6 if scenario == "S08 degradation scenario" and progress > 270 else 2

    factors = {
        "Inspect and service S08 now": (0.45, 0.60, 0.45),
        "Increase S08 effective capacity": (0.60, 0.72, 0.62),
        "Reduce upstream release rate": (0.78, 0.65, 0.78),
        "No intervention": (1.00, 1.00, 1.00)
    }
    rf,qf,ef = factors[intervention]

    new_risk = base_risk * rf
    new_queue = max(0, base_queue * qf)
    new_exposure = max(0, int(round(base_exposure * ef)))

    c1,c2,c3 = st.columns(3)
    c1.metric("Projected bottleneck risk", f"{new_risk:.0f}%", f"{new_risk-base_risk:.0f} pts")
    c2.metric("Projected S08 queue", f"{new_queue:.0f}", f"{new_queue-base_queue:.0f}")
    c3.metric("Projected downstream exposure", f"{new_exposure} stations")

    compare = pd.DataFrame({
        "Metric":["S08 risk (%)","Queue length","Downstream exposure (stations)"],
        "Current":[base_risk,base_queue,base_exposure],
        "Intervention scenario":[new_risk,new_queue,new_exposure]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=compare["Metric"], y=compare["Current"]))
    fig.add_trace(go.Bar(name="Intervention scenario", x=compare["Metric"], y=compare["Intervention scenario"]))
    fig.update_layout(barmode="group", height=390)
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "**TwinPulse workflow:** detect emerging risk → explain why → test intervention scenarios → "
        "recommend action with appropriate human oversight."
    )

with tabs[5]:
    st.subheader("Business Impact Pathway")
    st.caption(
        "Illustrative operational scenarios only. These figures are not claimed as live-factory results."
    )

    before = {"Downtime exposure":12, "Peak WIP":58, "Downstream stations exposed":6}
    after = {"Downtime exposure":5, "Peak WIP":42, "Downstream stations exposed":2}

    c1,c2,c3 = st.columns(3)
    c1.metric("Downtime exposure", f"{after['Downtime exposure']} min", "Earlier intervention scenario")
    c2.metric("Peak WIP", after["Peak WIP"], "Reduced propagation scenario")
    c3.metric("Downstream exposure", after["Downstream stations exposed"], "Stations affected")

    comp = pd.DataFrame({
        "Metric":list(before.keys()),
        "Before early warning":list(before.values()),
        "With early warning scenario":list(after.values())
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before early warning", x=comp["Metric"], y=comp["Before early warning"]))
    fig.add_trace(go.Bar(name="With early warning scenario", x=comp["Metric"], y=comp["With early warning scenario"]))
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Operational value chain:** Earlier signal → Earlier investigation → Targeted intervention → "
        "Less queue propagation → Lower rework/scrap exposure → Higher effective throughput."
    )

with st.expander("Technical mechanism implemented in TwinPulse V3"):
    st.write(
        "Station-specific baseline → deviation detection → combine available signals → transparent risk scoring → "
        "affected-station identification → connected production-line context → downstream exposure tracing → "
        "confidence-aware presentation → intervention scenario testing."
    )
