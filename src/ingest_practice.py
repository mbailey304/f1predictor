import sys
import fastf1
import pandas as pd

from config import CACHE_DIR, RAW_DIR, DEFAULT_PREDICT_YEAR, DEFAULT_PREDICT_ROUND

fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_practice_features(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    laps = laps.pick_quicklaps()
    if laps.empty:
        return pd.DataFrame()

    grouped = laps.groupby("Driver").agg(
        BestLapTime=("LapTime", "min"),
        AvgLapTime=("LapTime", "mean"),
        LapCount=("LapNumber", "count"),
        BestSector1=("Sector1Time", "min"),
        BestSector2=("Sector2Time", "min"),
        BestSector3=("Sector3Time", "min"),
    ).reset_index()

    for col in ["BestLapTime", "AvgLapTime", "BestSector1", "BestSector2", "BestSector3"]:
        grouped[col] = pd.to_timedelta(grouped[col], errors="coerce").dt.total_seconds()

    results = session.results[["Abbreviation", "FullName", "TeamName"]].copy()

    merged = results.merge(
        grouped,
        left_on="Abbreviation",
        right_on="Driver",
        how="left"
    )

    merged["Year"] = year
    merged["Round"] = round_number
    merged["EventName"] = event["EventName"]
    merged["SessionCode"] = session_code

    return merged[
        [
            "Year", "Round", "EventName", "SessionCode",
            "FullName", "TeamName",
            "BestLapTime", "AvgLapTime", "LapCount",
            "BestSector1", "BestSector2", "BestSector3"
        ]
    ]


def collect_practice_for_weekend(year: int, round_number: int) -> pd.DataFrame:
    rows = []

    for session_code in ["FP1", "FP2", "FP3"]:
        try:
            df = get_practice_features(year, round_number, session_code)
            if not df.empty:
                rows.append(df)
                print(f"Added {year} round {round_number} {session_code}")
        except Exception as e:
            print(f"Skipped {year} round {round_number} {session_code}: {e}")

    if not rows:
        print("No practice data collected.")
        return pd.DataFrame()

    final_df = pd.concat(rows, ignore_index=True)
    out_file = RAW_DIR / f"{year}_{round_number}_practice_features.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Saved practice file to {out_file}")
    return final_df


if __name__ == "__main__":
    year = DEFAULT_PREDICT_YEAR
    round_number = DEFAULT_PREDICT_ROUND

    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        round_number = int(sys.argv[2])

    collect_practice_for_weekend(year, round_number)