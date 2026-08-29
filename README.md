# TwinPulse

An adaptive digital twin prototype for a mixed-model vehicle assembly line,
built for the Accenture Innovation Challenge 2026 (DigitalTwin.ai track).

> **Note:** This project uses entirely SIMULATED / synthetic data. No real
> enterprise, company, or manufacturer data is used anywhere in this
> prototype.

## Project Status: Stage 1 — Foundation & Synthetic Data

This is the **first stage** of development. At this stage we have built:

- The project scaffolding.
- A reproducible synthetic production-data generator (`src/data_generator.py`).
- A validation script that checks the generated data behaves the way a
  realistic assembly line data would (`notebooks/validate_dataset.py`).

**Not built yet** (future stages): sensor-gap inference/imputation,
anomaly/defect prediction models, and the Streamlit dashboard (`app.py`
is currently a placeholder).

## The Simulated Line

- **30 stations**, in three phases:
  - `S01`–`S10`: Body Construction
  - `S11`–`S20`: Paint
  - `S21`–`S30`: Final Assembly
- Each station has a **sensor coverage level**: `HIGH`, `PARTIAL`, or
  `MANUAL` — sensor coverage is deliberately uneven across the line, just
  like a real patchwork plant.
- **400 vehicles** flow sequentially through all 30 stations, producing
  12,000 station-level observations.

## The Hidden Scenario (by design)

Station **S08** has a slow, gradual equipment degradation baked into the
data (rising cycle time, torque variability, vibration, temperature, and
queue length over the course of the simulated shift/run). It is **not** a
sudden failure — it ramps up smoothly.

Critically, S08 is a `PARTIAL` coverage station: its **vibration and
temperature sensors are missing** in the dataset, even though those are
exactly the signals that would most obviously reveal the problem. Only
`cycle_time`, `queue_length`, and `torque` are observable there. This is
intentional — a later stage of this project is expected to infer the
missing signals from what *is* observable, rather than reading them
directly.

The consequence of the Station 8 issue shows up downstream: the defect
rate at the **S22** inspection station rises noticeably as the run
progresses (roughly 4–5% early in the run vs. ~20–25% later), even though
nothing about S22 itself changed. This is the "signal to find" for later
analytics/ML work.

## Project Structure

```
TwinPulse/
├── data/
│   ├── production_data.csv   # generated vehicle x station observations
│   └── stations.csv          # station metadata
├── src/
│   └── data_generator.py     # the synthetic data generator (this stage)
├── notebooks/
│   └── validate_dataset.py   # validation checks + summary stats
├── models/                   # (empty for now — future ML stage)
├── app.py                    # placeholder for the future dashboard
├── requirements.txt
└── README.md
```

## How to Generate the Dataset

```bash
pip install -r requirements.txt
python src/data_generator.py
```

This writes `data/stations.csv` and `data/production_data.csv`. The
generator uses a fixed random seed (`RANDOM_SEED = 42` in
`src/data_generator.py`), so re-running it always produces an identical
dataset — this is verified automatically by the validation script.

## How to Verify the Dataset (including the hidden Station 8 scenario)

```bash
python notebooks/validate_dataset.py
```

This checks and prints evidence for:

1. There are exactly 30 stations.
2. Multiple vehicles (400) are represented, each passing through all 30
   stations.
3. Missing sensor values exist (structurally, based on each station's
   sensor coverage level).
4. **Station 8 shows gradual degradation over time** — cycle time, queue
   length, and torque variability all trend upward across the run, in a
   smooth ramp rather than a sudden jump.
5. Defects occur downstream, concentrated at the S22 inspection point,
   and the defect rate visibly rises as Station 8's condition worsens.
6. The dataset is reproducible: regenerating it produces a byte-identical
   CSV file.

The script prints a PASS/FAIL line for every check plus summary
statistics (per-column descriptive stats, defect counts by inspection
station, and the sensor coverage distribution).

## Columns in `production_data.csv`

| Column | Description |
|---|---|
| `vehicle_id` | Unique ID for each simulated vehicle |
| `timestamp` | Simulated time the vehicle reached this station |
| `station_id` | `S01`–`S30` |
| `station_type` | Body Construction / Paint / Final Assembly |
| `cycle_time` | Seconds spent at this station |
| `queue_length` | Vehicles waiting ahead of this station |
| `throughput` | Approx. vehicles/hour at this station at this time |
| `torque` | Physical sensor reading (may be missing) |
| `temperature` | Physical sensor reading (may be missing) |
| `vibration` | Physical sensor reading (may be missing) |
| `part_quality` | Incoming component quality score (0–1) |
| `operator_variation` | Human-driven variability factor |
| `sensor_coverage` | `HIGH` / `PARTIAL` / `MANUAL` for this station |
| `inspection_result` | `PASS`/`FAIL` at inspection stations, else blank |
| `defect` | Boolean, only meaningful at inspection stations |

## Design Notes / Assumptions

- Data is fully synthetic, generated with `numpy`'s reproducible random
  number generator — no real company or enterprise data was used or is
  required.
- Missing values are **not** imputed at this stage on purpose — sensor
  inference is planned as a distinct, later stage of the project.
- No predictions or ML outputs are hardcoded or faked anywhere in this
  stage; only the input dataset itself is generated.
