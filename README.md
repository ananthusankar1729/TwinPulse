# TwinPulse V3

Adaptive digital twin MVP for manufacturing lines.

## Run locally
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## V3 additions
- Connected 30-station production relationship map
- Uneven sensor coverage: HIGH / PARTIAL / MANUAL
- S08 missing-sensor challenge scenario
- Explainable bottleneck prediction
- Downstream propagation tracing
- Quality-risk proxy
- Interactive what-if intervention simulator
- Human-in-the-loop decision framing

## Important
All production data is synthetic and reproducible. The what-if simulator and business-impact figures are illustrative MVP scenarios, not live-factory performance claims.
