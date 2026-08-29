TwinPulse
Adaptive Digital Twin for Vehicle Assembly Lines
Accenture Innovation Challenge 2026 — Problem Track 4: DigitalTwin.ai
Problem
Modern vehicle assembly lines are made up of dozens of interdependent stations. A quiet degradation at one upstream station — a torque tool drifting, a bearing starting to vibrate — often stays invisible until it surfaces as a costly defect several stations downstream. By then, the causal trail is cold, sensors may have gaps, and operators are left reacting instead of preventing.
Solution
TwinPulse is a lightweight, explainable digital twin that:
Continuously scores every station for anomaly, bottleneck risk, and defect risk using transparent statistics (not black-box ML).
Detects when a sensor is missing and estimates the value using neighboring stations and rolling history — clearly labeling it AI Inferred with a confidence score, instead of pretending it's measured.
Traces a downstream defect backward through the line and identifies the most likely contributor station, with supporting evidence.
Lets an operator simulate an intervention ("what if we reduce cycle time here?") and see the projected effect on risk and throughput before committing to a real change.
Key Features
🏭 Live production flow view — 20 stations rendered as a flow with 🟢/🟡/🔴 status.
🔮 Predictive risk scoring — bottleneck risk & defect risk computed from real deviations in the synthetic dataset, not hardcoded.
🧩 Sensor-gap handling — vibration & temperature at the affected station are deliberately blanked and reconstructed, with a transparent, data-availability-based confidence score.
🔍 Root-cause tracing — ranks upstream stations by anomaly score and proximity to the downstream defect, surfacing a "likely contributor," never a "proven cause."
🧪 What-if simulation — compares three interventions (reduce cycle time / increase capacity / no action) against current state and recommends the best one.
Architecture
Synthetic Data Generator  →  Sensor Gap + Inference  →  Station-Level Analytics
        (NumPy/Pandas)         (neighbor + rolling avg)   (anomaly / bottleneck / defect risk)
                                                                    │
                                                                    ▼
                                                      Root-Cause Ranking (upstream trace)
                                                                    │
                                                                    ▼
                                                       What-If Simulation Engine
                                                                    │
                                                                    ▼
                                                        Streamlit Dashboard (4 tabs)
Everything runs in a single process — no database, no external API, no cloud service, and no connection to real machinery or PLCs.
Technology Stack
Python 3
Streamlit — dashboard UI
Pandas / NumPy — synthetic data generation and statistics
Plotly — interactive charts
No deep learning, no external APIs, no authentication, no Docker — by design, to keep the prototype fully self-contained and auditable.
Synthetic Data Assumptions
All data in this prototype is simulated, generated with a fixed random seed for reproducibility. It is illustrative only and does not represent any real plant, vehicle program, or supplier.
20 stations × 300 vehicles (6,000 station-visits).
Each station has its own baseline cycle time, queue length, throughput, torque, temperature, and vibration, with normal random noise.
Station 8 is seeded with a gradual degradation that ramps up over the course of the run: cycle time and queue length rise, torque becomes more variable, vibration and temperature increase, and its own throughput falls.
Downstream stations after Station 8 show a decaying throughput and queue effect proportional to how degraded Station 8 was for that vehicle.
Defects are injected probabilistically for vehicles that passed through Station 8 while it was degraded, surfacing at inspection points around Stations 15–20 — mirroring how real defects are often caught well after the true cause.
Vibration and temperature readings at Station 8 are deliberately removed to simulate a sensor outage, then reconstructed from neighboring stations (7 & 9) and a rolling historical average, each estimate carrying a confidence score based on how many reference signals were available and how well they agreed — with confidence reduced further as the underlying process visibly drifts, since a neighbor-based proxy becomes less reliable during active degradation.
How to Run
bash
pip install -r requirements.txt
streamlit run app.py
The app opens in your browser at http://localhost:8501.
Demo Flow
Land on the dashboard — top metrics already show an elevated bottleneck risk and an early-warning banner.
🏭 Digital Twin — see the 20-station flow; Station 8 is flagged 🔴. Select it to see measured vs. sensor-coverage detail and its degrading trend.
🔮 Predictions — see the bottleneck and defect risk cards, and the plain-language explanation of why (cycle time / queue / throughput deviations).
🔍 Root Cause — see the backward trace to the downstream defect, Station 8 highlighted as the likely contributor with a confidence score and supporting signals, plus its AI-inferred vibration and temperature values.
🧪 What-If — choose an intervention (reduce cycle time, increase capacity, or none), see the current-vs-simulated comparison table, and the recommended option.
The four tabs tell one continuous story — detection → prediction → explanation → action — rather than four disconnected features.
Limitations
All data is synthetic; no real plant, vehicle, or supplier data was used or referenced.
Risk scores and confidence values are produced by simple, transparent statistical rules chosen for explainability and speed of development, not by trained or validated machine-learning models.
The "likely contributor" identified by root-cause tracing is a statistical inference based on anomaly strength and proximity — it is not a certified or proven root cause.
The what-if simulation projects outcomes using simplified, deterministic adjustment rules; it does not model real physical or mechanical constraints of an actual line.
There is no live connection to sensors, PLCs, MES, or any other plant system.
Future Scope
Replace synthetic generation with ingestion from real (anonymized) plant telemetry, validated against actual production outcomes.
Upgrade anomaly detection and root-cause inference to calibrated statistical or ML models once sufficient labeled data is available, while preserving explainability.
Extend sensor-gap inference with more sophisticated imputation (e.g., multivariate regression, Kalman filtering) and quantify inferred-vs-measured accuracy against ground truth once available.
Add multi-line and multi-plant views, historical trend storage, and alerting/notification integration.
Expand the what-if engine to support combined and sequenced interventions, and to incorporate cost/ROI tradeoffs for recommendations.
