"""
TwinPulse - Anomaly Detection: Example / Validation Script
==============================================================

This script runs the anomaly detector (src/anomaly_detection.py) and
then does two things:

1. Prints a readable example of the detector's output, in the format
   requested in the brief (station, status, key signals).

2. Runs a SIMPLE VALIDATION of whether the detector actually found the
   known simulated degradation scenario -- WITHOUT letting the detector
   itself cheat by looking at anything the generator used internally.

   Important honesty note: this dataset is entirely SIMULATED. We
   already know (because WE wrote the generator) that a gradual
   equipment issue was injected starting roughly 20% of the way through
   the production run, originating at Station 8. That is the "known
   simulated degradation scenario" being validated here.

   To keep this a fair test, the check below only uses information any
   real plant would also have -- the order vehicles moved through the
   line -- to split the run into an "early" half and a "late" half, and
   checks whether flagged anomalies concentrate in the late half AND at
   Station 8 specifically. It does NOT read src/data_generator.py's
   internal degradation_level() function or any hidden ground-truth
   labels; there are none stored in production_data.csv.

   This is a sanity check on our own synthetic scenario, not a claim
   about real-world detection accuracy. No accuracy percentage is
   reported, because we have no real-world labeled data to measure
   accuracy against -- reporting one would be fabricated.
"""

import os
import pandas as pd

from anomaly_detection import run_anomaly_detection, RESULTS_PATH


def print_station_report(results, station_id):
    station_rows = results[results.station_id == station_id]
    print(f"\nStation {station_id}:")
    for _, row in station_rows.iterrows():
        if row.anomaly_status == "LOW":
            continue  # keep the printed report focused on what's interesting
        print(f"  Vehicles {int(row.window_start_vehicle):>3}-{int(row.window_end_vehicle):<3} "
              f"| score={row.anomaly_score:.2f} | Anomaly Status: {row.anomaly_status}")
        if row.key_signals != "none":
            print(f"      Key signals: {row.key_signals}")


def main():
    print("Running anomaly detection across all stations...\n")
    results = run_anomaly_detection()

    # -----------------------------------------------------------------
    # Part 1: Example output, in the requested format
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUT")
    print("=" * 70)
    print_station_report(results, "S08")

    # Also show a couple of ordinary, healthy stations for contrast --
    # picked generically (lowest max anomaly score), not cherry-picked.
    max_scores = results.groupby("station_id")["anomaly_score"].max().sort_values()
    quietest_stations = max_scores.index[:2].tolist()
    print(f"\nFor contrast, here are two of the quietest stations in the "
          f"whole run ({', '.join(quietest_stations)}):")
    for station_id in quietest_stations:
        print_station_report(results, station_id)
        station_all = results[results.station_id == station_id]
        if (station_all.anomaly_status == "LOW").all():
            print("  (every window stayed LOW for this station)")

    # -----------------------------------------------------------------
    # Part 2: Validation against the known simulated scenario
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION AGAINST THE KNOWN SIMULATED SCENARIO")
    print("=" * 70)
    print("Reminder: this entire dataset is SIMULATED. We are checking")
    print("whether the detector (using ONLY production_data.csv columns,")
    print("never any hidden generator internals) rediscovers the")
    print("degradation we know we injected around Station 8.\n")

    midpoint_vehicle = results["window_end_vehicle"].max() / 2

    early = results[results["window_end_vehicle"] <= midpoint_vehicle]
    late = results[results["window_start_vehicle"] > midpoint_vehicle]

    def high_rate(df):
        return (df["anomaly_status"] == "HIGH").mean()

    print(f"Share of ALL station-windows flagged HIGH, first half of the run:  {high_rate(early):.1%}")
    print(f"Share of ALL station-windows flagged HIGH, second half of the run: {high_rate(late):.1%}")

    # Rank every station by how anomalous its worst window was.
    ranking = (results.groupby("station_id")["anomaly_score"]
               .max()
               .sort_values(ascending=False))
    print("\nStations ranked by their single worst window score (top 5):")
    print(ranking.head(5).to_string())

    top_station = ranking.index[0]
    print(f"\n=> The station with the single worst anomaly score in the "
          f"entire simulated run is: {top_station}")
    if top_station == "S08":
        print("   This matches the station where the degradation was injected.")
    else:
        print("   NOTE: this does not match Station 8 -- if you're seeing this, "
              "the detector or the scenario may need review.")

    n_high_by_station = (results[results.anomaly_status == "HIGH"]
                          .groupby("station_id").size()
                          .sort_values(ascending=False))
    print("\nStations with at least one HIGH window (count of HIGH windows):")
    print(n_high_by_station.to_string() if len(n_high_by_station) else "  (none)")

    print("\nThis is a sanity check against a scenario we designed ourselves,")
    print("not a measured accuracy figure. It only shows that the detector's")
    print("output is consistent with the known injected scenario in this")
    print("particular synthetic dataset.")


if __name__ == "__main__":
    main()