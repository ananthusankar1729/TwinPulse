"""
TwinPulse - Synthetic Production Data Generator
================================================

This script generates a REALISTIC, SIMULATED dataset representing vehicles
moving through a 30-station mixed-model assembly line.

IMPORTANT: This is NOT real enterprise data. It is entirely synthetic and
generated for prototyping / competition purposes only.

What this script does
----------------------
1. Defines 30 stations (Body Construction -> Paint -> Final Assembly),
   each with a sensor coverage level (HIGH / PARTIAL / MANUAL).
2. Simulates ``NUM_VEHICLES`` vehicles moving sequentially through all
   30 stations, generating realistic process readings at every station.
3. Injects a HIDDEN, GRADUAL degradation trend that originates around
   Station 8 (equipment wear-style drift, not a sudden failure). This
   degradation slowly increases cycle time, torque variability,
   vibration, and temperature at Station 8, and increases queue length
   there. It also raises the probability of a defect being caught later
   at a downstream inspection station (Station 22).
4. Applies REALISTIC MISSING DATA based on each station's sensor
   coverage level. Some of the very signals that would reveal the
   Station 8 issue (vibration, temperature) are intentionally NOT
   captured at Station 8, mirroring a real partially-instrumented plant.
   We deliberately do NOT fill these gaps here -- that is left for a
   later "sensor inference" stage.
5. Writes two files:
     - data/stations.csv         (station metadata)
     - data/production_data.csv  (per-vehicle, per-station observations)

Reproducibility
----------------
A fixed random seed (``RANDOM_SEED``) is used everywhere, so re-running
this script always produces the exact same dataset.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUM_VEHICLES = 400          # "several hundred" vehicles -> 400 * 30 = 12,000 observations
TAKT_TIME_SECONDS = 90      # average time between vehicles entering the line
LINE_START_TIME = datetime(2026, 1, 5, 6, 0, 0)  # arbitrary Monday 06:00 shift start

STATION_TYPES = {
    **{f"S{i:02d}": "Body Construction" for i in range(1, 11)},
    **{f"S{i:02d}": "Paint" for i in range(11, 21)},
    **{f"S{i:02d}": "Final Assembly" for i in range(21, 31)},
}

# Baseline (normal / healthy) cycle time in seconds for each station.
# Varies a little by station type to feel realistic -- paint stations
# tend to run a bit longer than body/assembly stations, for example.
BASE_CYCLE_TIME = {}
for station_id, station_type in STATION_TYPES.items():
    idx = int(station_id[1:])
    if station_type == "Body Construction":
        BASE_CYCLE_TIME[station_id] = 70 + (idx % 5) * 3
    elif station_type == "Paint":
        BASE_CYCLE_TIME[station_id] = 85 + (idx % 5) * 4
    else:  # Final Assembly
        BASE_CYCLE_TIME[station_id] = 65 + (idx % 5) * 3

# Sensor coverage level per station. This is intentionally uneven, as
# described in the brief:
#   HIGH    -> torque, temperature, vibration all captured automatically
#   PARTIAL -> some of those signals captured, some are missing
#   MANUAL  -> no automated physical sensors; readings depend on manual
#              checklists (torque/temperature/vibration are NOT available)
SENSOR_COVERAGE = {
    # --- Body Construction (S01-S10) -------------------------------------
    "S01": "HIGH", "S02": "HIGH", "S03": "HIGH", "S04": "PARTIAL", "S05": "HIGH",
    "S06": "HIGH", "S07": "PARTIAL",
    "S08": "PARTIAL",   # <-- the hidden-degradation station (see brief)
    "S09": "HIGH", "S10": "HIGH",
    # --- Paint (S11-S20) --------------------------------------------------
    "S11": "HIGH", "S12": "PARTIAL", "S13": "PARTIAL", "S14": "MANUAL", "S15": "HIGH",
    "S16": "PARTIAL", "S17": "HIGH", "S18": "PARTIAL", "S19": "MANUAL", "S20": "HIGH",
    # --- Final Assembly (S21-S30) -----------------------------------------
    "S21": "MANUAL", "S22": "HIGH",  # S22 is a key inspection point -> well instrumented
    "S23": "MANUAL", "S24": "PARTIAL", "S25": "MANUAL", "S26": "MANUAL",
    "S27": "PARTIAL", "S28": "MANUAL", "S29": "MANUAL", "S30": "HIGH",
}

# For PARTIAL stations, which of the three physical sensors are actually
# available? (Anything not listed for a station is missing.) This gives
# every PARTIAL station its own distinct "missing data fingerprint",
# exactly as real plants tend to have.
PARTIAL_SENSOR_AVAILABILITY = {
    "S04": ["torque", "temperature"],          # vibration missing
    "S07": ["torque", "vibration"],             # temperature missing
    "S08": ["cycle_time_ok", "torque"],         # vibration AND temperature missing (per brief)
    "S12": ["temperature", "vibration"],        # torque missing
    "S13": ["torque"],                          # temperature + vibration missing
    "S16": ["vibration"],                       # torque + temperature missing
    "S18": ["torque", "temperature"],           # vibration missing
    "S24": ["temperature"],                     # torque + vibration missing
    "S27": ["torque", "vibration"],             # temperature missing
}

# Stations where a formal quality inspection happens (produces a real
# PASS/FAIL inspection_result). S22 is the key downstream inspection
# point that the Station 8 degradation is designed to eventually affect.
INSPECTION_STATIONS = {"S10", "S20", "S22", "S30"}

DEGRADATION_STATION = "S08"
DEGRADATION_START_PROGRESS = 0.20   # degradation is negligible before this point in the run
DEGRADATION_TARGET_STATION = "S22"  # downstream station where defect risk shows up

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_stations_table():
    """Build the static station metadata table."""
    rows = []
    for station_id in sorted(STATION_TYPES.keys(), key=lambda s: int(s[1:])):
        rows.append({
            "station_id": station_id,
            "station_type": STATION_TYPES[station_id],
            "normal_cycle_time": BASE_CYCLE_TIME[station_id],
            "sensor_coverage": SENSOR_COVERAGE[station_id],
        })
    return pd.DataFrame(rows)


def degradation_level(progress):
    """
    Returns a smooth, gradually-increasing degradation factor in [0, 1]
    as a function of how far through the simulated production run we are
    (progress = vehicle_index / NUM_VEHICLES).

    Degradation is ~0 for the first 20% of the run (healthy equipment),
    then ramps up in a smooth S-curve, mimicking gradual mechanical wear
    rather than a sudden failure.
    """
    if progress <= DEGRADATION_START_PROGRESS:
        return 0.0
    x = (progress - DEGRADATION_START_PROGRESS) / (1 - DEGRADATION_START_PROGRESS)
    # Smoothstep-like curve: gentle start, steeper middle, plateaus near 1.0
    return float(np.clip(3 * x**2 - 2 * x**3, 0, 1))


def generate_production_data(rng):
    """Generate the full per-vehicle, per-station observation dataset."""
    station_order = sorted(STATION_TYPES.keys(), key=lambda s: int(s[1:]))
    records = []

    for v_idx in range(1, NUM_VEHICLES + 1):
        vehicle_id = f"V{v_idx:05d}"
        progress = v_idx / NUM_VEHICLES
        deg = degradation_level(progress)  # how "worn" station 8 is right now

        vehicle_start_time = LINE_START_TIME + timedelta(seconds=v_idx * TAKT_TIME_SECONDS)
        elapsed_seconds = 0.0

        # This vehicle's "hidden quality debt" picked up if/when it passes
        # through the degraded Station 8. Carried forward to influence the
        # defect probability at the downstream inspection station.
        hidden_risk = 0.0

        for station_id in station_order:
            station_type = STATION_TYPES[station_id]
            coverage = SENSOR_COVERAGE[station_id]
            base_ct = BASE_CYCLE_TIME[station_id]

            is_degrading_station = (station_id == DEGRADATION_STATION)
            local_deg = deg if is_degrading_station else 0.0

            # --- Cycle time --------------------------------------------------
            noise_std = base_ct * 0.06
            cycle_time = rng.normal(base_ct * (1 + 0.35 * local_deg), noise_std * (1 + local_deg))
            cycle_time = max(cycle_time, base_ct * 0.5)

            # --- Queue length (vehicles waiting ahead of this station) ------
            base_queue = {"Body Construction": 2.0, "Paint": 3.0, "Final Assembly": 2.5}[station_type]
            # Small ripple effect: stations right after S08 feel a bit of the backup
            distance_from_deg = max(0, int(station_id[1:]) - int(DEGRADATION_STATION[1:]))
            ripple = deg * max(0, 1 - distance_from_deg * 0.25) if int(station_id[1:]) >= int(DEGRADATION_STATION[1:]) else 0
            queue_length = max(0, rng.normal(base_queue + 6 * local_deg + 1.5 * ripple, 0.8))

            # --- Throughput (vehicles/hour equivalent at this station) ------
            theoretical_throughput = 3600.0 / base_ct
            throughput = theoretical_throughput * (1 - 0.3 * local_deg - 0.1 * ripple)
            throughput = max(throughput * rng.normal(1.0, 0.03), 1.0)

            # --- Physical sensor readings (torque / temperature / vibration) -
            base_torque = 100 + rng.normal(0, 3)
            torque = rng.normal(base_torque * (1 + 0.05 * local_deg), 2.5 * (1 + 1.5 * local_deg))

            base_temp = 24 if station_type != "Paint" else 60  # paint booths run hot
            temperature = rng.normal(base_temp + 6 * local_deg, 1.2 * (1 + local_deg))

            base_vibration = 0.5
            vibration = max(0.0, rng.normal(base_vibration + 0.9 * local_deg, 0.1 * (1 + 2 * local_deg)))

            # --- Part quality (incoming component quality score, 0-1) -------
            part_quality = float(np.clip(rng.normal(0.93, 0.04), 0, 1))

            # --- Operator variation (human-driven inconsistency factor) -----
            operator_sigma = {"HIGH": 0.03, "PARTIAL": 0.06, "MANUAL": 0.10}[coverage]
            operator_variation = float(rng.normal(0, operator_sigma))

            # --- Accumulate hidden risk from the degrading station ----------
            if is_degrading_station:
                hidden_risk = local_deg  # snapshot of how worn the line was when this vehicle passed through

            # --- Defect / inspection outcome ---------------------------------
            defect = False
            inspection_result = None
            if station_id in INSPECTION_STATIONS:
                base_defect_rate = 0.02
                if station_id == DEGRADATION_TARGET_STATION:
                    # Logistic-style bump: higher hidden_risk -> higher defect probability,
                    # but still probabilistic/noisy, not a hard rule.
                    extra_risk = 0.35 / (1 + np.exp(-8 * (hidden_risk - 0.5)))
                    defect_prob = base_defect_rate + extra_risk
                else:
                    defect_prob = base_defect_rate
                defect = bool(rng.random() < defect_prob)
                inspection_result = "FAIL" if defect else "PASS"

            timestamp = vehicle_start_time + timedelta(seconds=elapsed_seconds)
            elapsed_seconds += cycle_time

            records.append({
                "vehicle_id": vehicle_id,
                "timestamp": timestamp,
                "station_id": station_id,
                "station_type": station_type,
                "cycle_time": round(cycle_time, 2),
                "queue_length": round(queue_length, 2),
                "throughput": round(throughput, 2),
                "torque": round(torque, 2),
                "temperature": round(temperature, 2),
                "vibration": round(vibration, 3),
                "part_quality": round(part_quality, 3),
                "operator_variation": round(operator_variation, 3),
                "sensor_coverage": coverage,
                "inspection_result": inspection_result,
                "defect": defect,
            })

    return pd.DataFrame(records)


def apply_sensor_gaps(df, rng):
    """
    Mask out sensor columns that a station's sensor_coverage level does not
    actually capture, plus a small amount of random real-world sensor
    dropout on top. We do NOT fill any of these gaps here.
    """
    df = df.copy()
    sensor_cols = ["torque", "temperature", "vibration"]

    for station_id, coverage in SENSOR_COVERAGE.items():
        station_mask = df["station_id"] == station_id

        if coverage == "HIGH":
            available = set(sensor_cols)
        elif coverage == "MANUAL":
            available = set()  # no automated physical sensors at all
        else:  # PARTIAL
            configured = PARTIAL_SENSOR_AVAILABILITY.get(station_id, [])
            available = set(c for c in configured if c in sensor_cols)

        for col in sensor_cols:
            if col not in available:
                df.loc[station_mask, col] = np.nan

    # Small amount of extra random dropout (~2%) on whatever sensor values
    # ARE supposed to be available, to mimic occasional real-world sensor
    # glitches unrelated to structural coverage gaps.
    for col in sensor_cols:
        available_mask = df[col].notna()
        dropout_mask = available_mask & (rng.random(len(df)) < 0.02)
        df.loc[dropout_mask, col] = np.nan

    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stations_df = build_stations_table()
    production_df = generate_production_data(rng)
    production_df = apply_sensor_gaps(production_df, rng)

    stations_path = os.path.join(OUTPUT_DIR, "stations.csv")
    production_path = os.path.join(OUTPUT_DIR, "production_data.csv")

    stations_df.to_csv(stations_path, index=False)
    production_df.to_csv(production_path, index=False)

    print(f"Generated {len(stations_df)} stations -> {stations_path}")
    print(f"Generated {len(production_df)} observations across {production_df['vehicle_id'].nunique()} vehicles -> {production_path}")


if __name__ == "__main__":
    main()
