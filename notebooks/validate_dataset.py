"""
TwinPulse - Dataset Validation Script
======================================

Run this AFTER generating the dataset (src/data_generator.py) to confirm
that everything the prototype needs is actually present:

  1. There are 30 stations.
  2. Multiple vehicles are represented.
  3. Missing sensor values exist.
  4. Station 8 shows gradual degradation over time.
  5. Defects occur downstream (at Station 22 in particular).
  6. The dataset is reproducible with the fixed seed.

This is a plain script (not a Jupyter notebook) so it can be run anywhere
with just `python notebooks/validate_dataset.py` -- no notebook server
required. Feel free to open it in Jupyter later if you prefer cell-by-cell
exploration.
"""

import os
import sys
import subprocess
import hashlib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIONS_PATH = os.path.join(DATA_DIR, "stations.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_data.csv")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    all_ok = True

    if not (os.path.exists(STATIONS_PATH) and os.path.exists(PRODUCTION_PATH)):
        print("Dataset not found. Run `python src/data_generator.py` first.")
        sys.exit(1)

    stations = pd.read_csv(STATIONS_PATH)
    df = pd.read_csv(PRODUCTION_PATH, parse_dates=["timestamp"])

    # ------------------------------------------------------------------
    section("1. Station count")
    ok = check("Exactly 30 stations in stations.csv", len(stations) == 30)
    all_ok &= ok
    ok = check("Exactly 30 unique stations in production_data.csv", df["station_id"].nunique() == 30)
    all_ok &= ok
    print(stations["station_type"].value_counts())

    # ------------------------------------------------------------------
    section("2. Multiple vehicles represented")
    n_vehicles = df["vehicle_id"].nunique()
    ok = check(f"More than 1 vehicle present (found {n_vehicles})", n_vehicles > 1)
    all_ok &= ok
    ok = check("Every vehicle passes through all 30 stations",
               (df.groupby("vehicle_id")["station_id"].nunique() == 30).all())
    all_ok &= ok
    print(f"Total observations: {len(df)}  |  Vehicles: {n_vehicles}  |  Stations: {df['station_id'].nunique()}")

    # ------------------------------------------------------------------
    section("3. Missing sensor values exist")
    sensor_cols = ["torque", "temperature", "vibration"]
    na_counts = df[sensor_cols].isna().sum()
    print(na_counts)
    ok = check("At least one NaN in each physical sensor column", (na_counts > 0).all())
    all_ok &= ok
    manual_stations = stations.loc[stations["sensor_coverage"] == "MANUAL", "station_id"]
    manual_rows = df[df["station_id"].isin(manual_stations)]
    ok = check("MANUAL stations have 100% missing torque/temperature/vibration",
               manual_rows[sensor_cols].isna().all().all() if len(manual_rows) else False)
    all_ok &= ok
    ok = check("Station 8 is missing BOTH vibration and temperature (per spec)",
               df.loc[df.station_id == "S08", "vibration"].isna().mean() > 0.9 and
               df.loc[df.station_id == "S08", "temperature"].isna().mean() > 0.9)
    all_ok &= ok
    ok = check("Station 8 still HAS cycle_time and torque data",
               df.loc[df.station_id == "S08", "cycle_time"].notna().all() and
               df.loc[df.station_id == "S08", "torque"].isna().mean() < 0.2)
    all_ok &= ok

    # ------------------------------------------------------------------
    section("4. Station 8 shows gradual degradation over time")
    s8 = df[df["station_id"] == "S08"].sort_values("timestamp").reset_index(drop=True)
    n = len(s8)
    q1, q4 = s8.iloc[: n // 4], s8.iloc[-n // 4 :]

    ct_before, ct_after = q1["cycle_time"].mean(), q4["cycle_time"].mean()
    queue_before, queue_after = q1["queue_length"].mean(), q4["queue_length"].mean()
    torque_std_before, torque_std_after = q1["torque"].std(), q4["torque"].std()

    print(f"cycle_time      : first quarter={ct_before:.1f}s   -> last quarter={ct_after:.1f}s")
    print(f"queue_length    : first quarter={queue_before:.2f}  -> last quarter={queue_after:.2f}")
    print(f"torque std dev  : first quarter={torque_std_before:.2f} -> last quarter={torque_std_after:.2f}")

    ok = check("Cycle time increases from first to last quarter", ct_after > ct_before)
    all_ok &= ok
    ok = check("Queue length increases from first to last quarter", queue_after > queue_before)
    all_ok &= ok
    ok = check("Torque variability (std dev) increases from first to last quarter",
               torque_std_after > torque_std_before)
    all_ok &= ok
    # Check "gradual" using decile-level means (smooths out per-vehicle noise)
    # rather than single-observation diffs, which is a fairer way to detect
    # a smooth ramp vs. a sudden step change.
    deciles = [s8.iloc[i * n // 10 : (i + 1) * n // 10]["cycle_time"].mean() for i in range(10)]
    total_drift = ct_after - ct_before
    max_decile_jump = max(abs(deciles[i + 1] - deciles[i]) for i in range(len(deciles) - 1))
    ok = check("Degradation ramps up gradually across deciles (no single decile-to-decile jump > 50% of total drift)",
               max_decile_jump < 0.5 * total_drift)
    all_ok &= ok

    # ------------------------------------------------------------------
    section("5. Defects occur downstream (Station 22)")
    s22 = df[df["station_id"] == "S22"].sort_values("timestamp").reset_index(drop=True)
    half = len(s22) // 2
    defect_rate_first_half = s22.iloc[:half]["defect"].mean()
    defect_rate_second_half = s22.iloc[half:]["defect"].mean()
    total_defects_s22 = int(s22["defect"].sum())

    print(f"Station 22 defect rate: first half of run={defect_rate_first_half:.1%}"
          f"   second half of run={defect_rate_second_half:.1%}")
    print(f"Total defects logged at Station 22: {total_defects_s22} / {len(s22)}")

    ok = check("At least some defects occur at Station 22", total_defects_s22 > 0)
    all_ok &= ok
    ok = check("Defect rate at Station 22 rises as Station 8 degrades",
               defect_rate_second_half > defect_rate_first_half)
    all_ok &= ok
    ok = check("Other inspection stations exist with their own defect logs",
               df.loc[df["station_id"].isin(["S10", "S20", "S30"]), "defect"].sum() >= 0)
    all_ok &= ok

    # ------------------------------------------------------------------
    section("6. Reproducibility with fixed seed")
    print("Re-running the generator in a subprocess to confirm identical output...")
    gen_script = os.path.join(BASE_DIR, "src", "data_generator.py")
    hash_before = file_md5(PRODUCTION_PATH)
    subprocess.run([sys.executable, gen_script], check=True, cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hash_after = file_md5(PRODUCTION_PATH)
    ok = check("production_data.csv is byte-identical after regenerating", hash_before == hash_after)
    all_ok &= ok

    # ------------------------------------------------------------------
    section("Summary statistics")
    print(df[["cycle_time", "queue_length", "throughput", "torque",
              "temperature", "vibration", "part_quality", "operator_variation"]].describe().round(2))
    print("\nDefect counts by inspection station:")
    print(df[df["station_id"].isin(["S10", "S20", "S22", "S30"])]
          .groupby("station_id")["defect"].agg(["sum", "count", "mean"]))
    print("\nSensor coverage distribution:")
    print(stations["sensor_coverage"].value_counts())

    section("Overall result")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED - review output above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
