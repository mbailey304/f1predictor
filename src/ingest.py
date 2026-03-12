import fastf1
import pandas as pd

from config import CACHE_DIR, RAW_DIR, START_YEAR, END_YEAR, SESSION_CODES

fastf1.Cache.enable_cache(str(CACHE_DIR))

PARTIAL_FILE = RAW_DIR / "historical_session_results_partial.csv"
FINAL_FILE = RAW_DIR / "historical_session_results.csv"


def get_session_results(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_code)

    # Load only what is needed for session results
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    df = session.results.copy()
    df["Year"] = year
    df["Round"] = round_number
    df["EventName"] = event["EventName"]
    df["SessionCode"] = session_code

    keep_cols = [
        "DriverNumber",
        "FullName",
        "Abbreviation",
        "TeamName",
        "Position",
        "ClassifiedPosition",
        "GridPosition",
        "Q1",
        "Q2",
        "Q3",
        "Time",
        "Status",
        "Points",
        "Laps",
        "Year",
        "Round",
        "EventName",
        "SessionCode",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def load_existing_partial() -> pd.DataFrame:
    if PARTIAL_FILE.exists():
        try:
            df = pd.read_csv(PARTIAL_FILE)
            print(f"Loaded partial progress: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Could not read partial file, starting fresh: {e}")
    return pd.DataFrame()


def already_done(existing_df: pd.DataFrame, year: int, round_number: int, session_code: str) -> bool:
    if existing_df.empty:
        return False

    needed = {"Year", "Round", "SessionCode"}
    if not needed.issubset(existing_df.columns):
        return False

    mask = (
        (existing_df["Year"] == year)
        & (existing_df["Round"] == round_number)
        & (existing_df["SessionCode"] == session_code)
    )
    return mask.any()


def save_progress(df: pd.DataFrame, filepath) -> None:
    df.to_csv(filepath, index=False)


def collect_all_results() -> pd.DataFrame:
    existing_df = load_existing_partial()
    rows = []

    if not existing_df.empty:
        rows.append(existing_df)

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Loading season {year}")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Skipped season {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            if event.is_testing():
                continue

            round_number = int(event["RoundNumber"])

            for session_code in SESSION_CODES:
                if already_done(existing_df, year, round_number, session_code):
                    print(f"  already saved {year} round {round_number} {session_code}")
                    continue

                try:
                    df = get_session_results(year, round_number, session_code)
                    if not df.empty:
                        rows.append(df)
                        existing_df = pd.concat(rows, ignore_index=True)

                        save_progress(existing_df, PARTIAL_FILE)
                        print(f"  added and saved {year} round {round_number} {session_code}")
                except Exception as e:
                    print(f"  skipped {year} round {round_number} {session_code}: {e}")

    if rows:
        final_df = pd.concat(rows, ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=["Year", "Round", "SessionCode", "FullName"],
            keep="last"
        ).reset_index(drop=True)

        save_progress(final_df, FINAL_FILE)
        print(f"Saved final historical file to {FINAL_FILE}")
        return final_df

    return pd.DataFrame()


if __name__ == "__main__":
    df = collect_all_results()
    print(f"Done. Total rows: {len(df)}")