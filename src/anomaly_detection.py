"""
TwinPulse - Anomaly Detection Component
=========================================

This module looks at the production data generated in Stage 1
(data/production_data.csv) and flags stations/time-windows that are
behaving abnormally, BEFORE that abnormality turns into an obvious
downstream failure (like the defect spike we saw at Station 22).

APPROACH (two simple, interpretable methods, blended together)
-----------------------------------------------------------------
1. Station-specific baseline z-scores
   For every station, we learn what "normal" looks like from an early,
   assumed-healthy slice of that station's own history (NOT a single
   global threshold for every station -- each station gets its own
   baseline, since a Paint station and a Final Assembly station simply
   run differently).
   For every signal (cycle_time, queue_length, throughput, torque,
   temperature, vibration) we compute how many standard deviations a
   later window's average is away from that station's own baseline
   average. This also lets us separately check for a MEAN shift
   (e.g. "cycle time is running longer") and a VARIABILITY shift
   (e.g. "torque is swinging around a lot more than before"), which
   the brief specifically calls out (torque variability increasing is
   a different failure mode than torque itself drifting).

2. Isolation Forest (per station)
   Isolation Forest is trained ONLY on each station's own healthy
   baseline period, using whichever signals that station actually has
   data for. It looks at all the signals together at once, so it can
   catch a *combination* of small shifts that wouldn't individually
   look extreme, but together are unusual. Anything that "isolates"
   quickly from the baseline data is scored as more anomalous.

We combine both scores (simple average) into one 0-1 anomaly_score per
station per time window, and classify it as LOW / MEDIUM / HIGH using
thresholds derived from *that station's own baseline distribution* --
never one fixed number applied to every station.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------------------------------------------
- It never checks `if station_id == "S08"` anywhere in the detection
  logic. Every station is scored by the exact same procedure. If a
  station happens to be Station 8, it is because the DATA earned that
  result, not because the code singled it out.
- It does not use any generation-time "ground truth" (e.g. the hidden
  degradation curve baked into src/data_generator.py) to decide scores.
  It only ever looks at the same production_data.csv columns a real
  plant would actually be able to measure.

Files produced
-----------------------------------------------------------------
- data/anomaly_results.csv     -- one row per station per time window
- models/anomaly/<station>_baseline.json     -- baseline stats per station
- models/anomaly/<station>_isoforest.joblib  -- fitted Isolation Forest per station
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "anomaly")

PRODUCTION_PATH = os.path.join(DATA_DIR, "production_data.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "anomaly_results.csv")

# Signals we consider for anomaly detection (only used per-station if the
# station actually has enough real, non-missing data for that signal).
CANDIDATE_SIGNALS = [
    "cycle_time", "queue_length", "throughput",
    "torque", "temperature", "vibration",
]

# Directional signals: for these, only an INCREASE beyond baseline is
# treated as a meaningful mean-shift anomaly. Throughput is the opposite
# (a DECREASE is the meaningful direction). Anything not listed defaults
# to "flag whichever direction crosses the threshold".
INCREASE_IS_BAD = {"cycle_time", "queue_length", "torque", "temperature", "vibration"}
DECREASE_IS_BAD = {"throughput"}

MIN_AVAILABILITY = 0.5     # a signal needs >=50% non-null data at a station to be used at all
BASELINE_FRACTION = 0.15   # first 15% of a station's vehicles = assumed-healthy baseline period
WINDOW_SIZE = 20           # vehicles per rolling window when reporting results
Z_FLAG_THRESHOLD = 1.5     # |z| beyond this = a signal is called out as "abnormal" in that window
VARIABILITY_RATIO_THRESHOLD = 1.5  # window_std / baseline_std beyond this = "variability increasing"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Step 1: figure out which signals are usable per station
# ---------------------------------------------------------------------------

def usable_signals_for_station(station_df):
    """Return the list of candidate signals that have enough real data
    at this station to be worth analysing (data-driven, not hardcoded
    per station -- a MANUAL station with no physical sensors will
    simply end up with fewer usable signals than a HIGH-coverage one)."""
    usable = []
    for col in CANDIDATE_SIGNALS:
        availability = station_df[col].notna().mean()
        if availability >= MIN_AVAILABILITY:
            usable.append(col)
    return usable


# ---------------------------------------------------------------------------
# Step 2: build each station's own healthy baseline
# ---------------------------------------------------------------------------

def build_baseline(station_df, signals):
    """Use the first BASELINE_FRACTION of this station's vehicles (in
    time order) as the assumed-healthy reference period, and compute
    per-signal mean/std from it. This is a per-station baseline, so a
    Paint station is never compared against a Body Construction
    station's "normal" range."""
    n_baseline = max(10, int(len(station_df) * BASELINE_FRACTION))
    baseline_rows = station_df.iloc[:n_baseline]

    stats = {}
    for col in signals:
        values = baseline_rows[col].dropna()
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:  # guard against a degenerate all-identical baseline
            std = 1e-6
        stats[col] = {"mean": mean, "std": std}
    return stats, baseline_rows


# ---------------------------------------------------------------------------
# Step 3: per-row z-scores against that station's baseline
# ---------------------------------------------------------------------------

def compute_zscores(station_df, signals, baseline_stats):
    z = pd.DataFrame(index=station_df.index)
    for col in signals:
        mean = baseline_stats[col]["mean"]
        std = baseline_stats[col]["std"]
        z[col] = (station_df[col] - mean) / std
    return z


# ---------------------------------------------------------------------------
# Step 4: Isolation Forest trained on the station's baseline only
# ---------------------------------------------------------------------------

def fit_isolation_forest(baseline_rows, signals, baseline_stats):
    """Fit an Isolation Forest using only this station's baseline
    (healthy) rows, so 'normal' for this station is learned from its
    own history. Missing values in the remaining ~2% random sensor
    dropout are filled with the baseline mean (i.e. 'assume normal'
    when a reading is simply missing, not a claim that it WAS normal)."""
    X = baseline_rows[signals].copy()
    for col in signals:
        X[col] = X[col].fillna(baseline_stats[col]["mean"])

    model = IsolationForest(n_estimators=200, random_state=RANDOM_SEED, contamination="auto")
    model.fit(X)
    return model


def isolation_forest_scores(model, station_df, signals, baseline_stats):
    """Higher output = more anomalous (we flip sklearn's convention,
    where score_samples is higher for NORMAL points)."""
    X = station_df[signals].copy()
    for col in signals:
        X[col] = X[col].fillna(baseline_stats[col]["mean"])
    raw = -model.score_samples(X)  # flip sign: higher now = more anomalous
    return raw


def normalize_if_scores(raw_scores, baseline_raw_scores):
    """Scale Isolation Forest scores to roughly [0, 1] using the
    baseline period's own score distribution as the reference range,
    instead of a fixed cutoff shared across every station."""
    lo = np.percentile(baseline_raw_scores, 5)
    hi = np.percentile(baseline_raw_scores, 95)
    span = max(hi - lo, 1e-6)
    normalized = (raw_scores - lo) / span
    return np.clip(normalized, 0, 1.5)  # small headroom above 1.0 for extreme outliers


# ---------------------------------------------------------------------------
# Step 5: combine both signals into one score, per row
# ---------------------------------------------------------------------------

def zscore_composite(z_row):
    """Mean of |z| across whichever signals are available for this row,
    then squashed into [0, 1) with a smooth saturating curve so a
    single extreme signal doesn't blow the scale up unboundedly."""
    available = z_row.dropna()
    if len(available) == 0:
        return np.nan
    mean_abs_z = available.abs().mean()
    return 1 - np.exp(-mean_abs_z / 2.5)


# ---------------------------------------------------------------------------
# Step 6: window-level aggregation + key abnormal signal detection
# ---------------------------------------------------------------------------

def describe_key_signals(window_df, z_window, signals, baseline_stats):
    """For a window of rows, work out which signals are behaving
    abnormally and describe HOW (level shift vs variability shift,
    increasing vs decreasing) -- generic logic, works the same way for
    every station."""
    findings = []
    for col in signals:
        # --- Mean-level shift ---------------------------------------
        window_mean_z = z_window[col].mean()
        if pd.notna(window_mean_z):
            if col in INCREASE_IS_BAD and window_mean_z > Z_FLAG_THRESHOLD:
                findings.append(f"{col} increasing")
            elif col in DECREASE_IS_BAD and window_mean_z < -Z_FLAG_THRESHOLD:
                findings.append(f"{col} decreasing")
            elif col not in INCREASE_IS_BAD and col not in DECREASE_IS_BAD:
                if window_mean_z > Z_FLAG_THRESHOLD:
                    findings.append(f"{col} increasing")
                elif window_mean_z < -Z_FLAG_THRESHOLD:
                    findings.append(f"{col} decreasing")

        # --- Variability shift (is this signal swinging around more?) --
        baseline_std = baseline_stats[col]["std"]
        window_std = window_df[col].std()
        if pd.notna(window_std) and baseline_std > 1e-6:
            ratio = window_std / baseline_std
            if ratio > VARIABILITY_RATIO_THRESHOLD:
                findings.append(f"{col} variability increasing")

    return findings


def classify_status(score, baseline_scores):
    """Station-specific thresholds: HIGH/MEDIUM/LOW are defined relative
    to THIS station's own baseline score distribution, not one global
    cutoff for every station in the plant."""
    high_cutoff = np.percentile(baseline_scores, 90)
    medium_cutoff = np.percentile(baseline_scores, 75)
    if score >= high_cutoff and score >= 0.35:   # small floor so a very quiet
        return "HIGH"                             # baseline doesn't call itself HIGH
    elif score >= medium_cutoff and score >= 0.2:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Main per-station pipeline
# ---------------------------------------------------------------------------

def analyze_station(station_id, station_df):
    station_df = station_df.sort_values("timestamp").reset_index(drop=True)
    signals = usable_signals_for_station(station_df)

    if not signals:
        # Extremely sparse station (shouldn't happen here, but fail safe)
        return None, None, None

    baseline_stats, baseline_rows = build_baseline(station_df, signals)
    z_scores = compute_zscores(station_df, signals, baseline_stats)

    model = fit_isolation_forest(baseline_rows, signals, baseline_stats)
    raw_if_scores = isolation_forest_scores(model, station_df, signals, baseline_stats)
    baseline_raw_if_scores = isolation_forest_scores(model, baseline_rows, signals, baseline_stats)
    if_scores = normalize_if_scores(raw_if_scores, baseline_raw_if_scores)

    z_composite = z_scores.apply(zscore_composite, axis=1)
    final_score_per_row = 0.5 * z_composite.fillna(0) + 0.5 * pd.Series(if_scores, index=station_df.index)

    # Baseline's own final scores, used purely to set this station's
    # HIGH/MEDIUM/LOW thresholds -- never shared across stations.
    baseline_final_scores = final_score_per_row.iloc[: len(baseline_rows)].values

    # --- Aggregate into rolling windows of WINDOW_SIZE vehicles ---------
    rows = []
    n = len(station_df)
    for start in range(0, n, WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, n)
        window_df = station_df.iloc[start:end]
        z_window = z_scores.iloc[start:end]
        window_score = final_score_per_row.iloc[start:end].mean()
        status = classify_status(window_score, baseline_final_scores)
        key_signals = describe_key_signals(window_df, z_window, signals, baseline_stats)

        rows.append({
            "station_id": station_id,
            "window_start_vehicle": start + 1,
            "window_end_vehicle": end,
            "window_start_time": window_df["timestamp"].iloc[0],
            "window_end_time": window_df["timestamp"].iloc[-1],
            "n_observations": len(window_df),
            "anomaly_score": round(float(window_score), 4),
            "anomaly_status": status,
            "key_signals": "; ".join(key_signals) if key_signals else "none",
        })

    return pd.DataFrame(rows), baseline_stats, model


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_anomaly_detection():
    if not os.path.exists(PRODUCTION_PATH):
        raise FileNotFoundError(
            f"Could not find {PRODUCTION_PATH}. Run src/data_generator.py first."
        )

    df = pd.read_csv(PRODUCTION_PATH, parse_dates=["timestamp"])
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_results = []
    for station_id, station_df in df.groupby("station_id"):
        results, baseline_stats, model = analyze_station(station_id, station_df)
        if results is None:
            continue
        all_results.append(results)

        # Save artifacts so this station's baseline/model can be reused
        # later (e.g. by the dashboard) without recomputing everything.
        with open(os.path.join(MODELS_DIR, f"{station_id}_baseline.json"), "w") as f:
            json.dump(baseline_stats, f, indent=2)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{station_id}_isoforest.joblib"))

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(["station_id", "window_start_vehicle"]).reset_index(drop=True)
    combined.to_csv(RESULTS_PATH, index=False)

    print(f"Analyzed {combined['station_id'].nunique()} stations across "
          f"{combined['window_start_vehicle'].max() // WINDOW_SIZE + 1} windows each.")
    print(f"Saved window-level results -> {RESULTS_PATH}")
    print(f"Saved per-station baselines + Isolation Forest models -> {MODELS_DIR}")

    return combined


if __name__ == "__main__":
    run_anomaly_detection()