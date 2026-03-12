import sys
import numpy as np
import pandas as pd
import joblib
import fastf1

from config import (
    CACHE_DIR,
    RAW_DIR,
    OUTPUTS_DIR,
    CURRENT_SEASON,
    DEFAULT_PREDICT_YEAR,
    DEFAULT_PREDICT_ROUND,
)
from features import (
    clean_results,
    add_form_features,
    add_current_season_features,
    encode_features,
)
from utils import get_prior_weight, get_current_weight

fastf1.Cache.enable_cache(str(CACHE_DIR))


def build_prediction_rows(
    history_df: pd.DataFrame,
    year: int,
    round_number: int,
    session_code: str
) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    event_row = schedule[schedule["RoundNumber"] == round_number].iloc[0]
    event_name = event_row["EventName"]

    event = fastf1.get_event(year, round_number)

    current_drivers = None
    for session_try in ["R", "Q", "SQ", "S"]:
        try:
            s = event.get_session(session_try)
            s.load(laps=False, telemetry=False, weather=False, messages=False)
            current_drivers = s.results[["FullName", "TeamName"]].copy()
            break
        except Exception:
            continue

    if current_drivers is None or current_drivers.empty:
        prior = history_df[
            (history_df["Year"] == year) & (history_df["Round"] == round_number - 1)
        ]
        current_drivers = prior[["FullName", "TeamName"]].drop_duplicates().copy()

    current_drivers = current_drivers.drop_duplicates(
        subset=["FullName", "TeamName"]
    ).reset_index(drop=True)

    pred_df = current_drivers.copy()
    pred_df["Year"] = year
    pred_df["Round"] = round_number
    pred_df["EventName"] = event_name
    pred_df["SessionCode"] = session_code

    pred_df["Position"] = np.nan
    pred_df["GridPosition"] = np.nan
    pred_df["Points"] = np.nan
    pred_df["Laps"] = np.nan

    combined = pd.concat([history_df, pred_df], ignore_index=True)
    combined = clean_results(combined)
    combined = add_form_features(combined)
    combined = add_current_season_features(combined)

    target_rows = combined[
        (combined["Year"] == year)
        & (combined["Round"] == round_number)
        & (combined["SessionCode"] == session_code)
    ].copy()

    target_rows = target_rows.drop_duplicates(
        subset=["FullName", "TeamName"],
        keep="last"
    ).reset_index(drop=True)

    return target_rows


def attach_practice_features(
    pred_rows: pd.DataFrame,
    year: int,
    round_number: int
) -> pd.DataFrame:
    path = RAW_DIR / f"{year}_{round_number}_practice_features.csv"
    if not path.exists():
        return pred_rows

    practice = pd.read_csv(path)

    practice = practice.drop_duplicates(
        subset=["Year", "Round", "EventName", "FullName", "TeamName", "SessionCode"],
        keep="last"
    ).copy()

    agg = (
        practice.groupby(["Year", "Round", "EventName", "FullName", "TeamName"])
        .agg(
            PracticeBestLap=("BestLapTime", "min"),
            PracticeAvgLap=("AvgLapTime", "mean"),
            PracticeBestS1=("BestSector1", "min"),
            PracticeBestS2=("BestSector2", "min"),
            PracticeBestS3=("BestSector3", "min"),
            PracticeLapCount=("LapCount", "sum"),
        )
        .reset_index()
    )

    merged = pred_rows.merge(
        agg,
        on=["Year", "Round", "EventName", "FullName", "TeamName"],
        how="left"
    )

    return merged


def attach_current_weekend_results(
    pred_rows: pd.DataFrame,
    year: int,
    round_number: int
) -> pd.DataFrame:
    history_file = RAW_DIR / "historical_session_results.csv"
    if not history_file.exists():
        return pred_rows

    hist = pd.read_csv(history_file)

    same_weekend = hist[
        (hist["Year"] == year) &
        (hist["Round"] == round_number)
    ].copy()

    if same_weekend.empty:
        return pred_rows

    same_weekend["Position"] = pd.to_numeric(same_weekend["Position"], errors="coerce")
    same_weekend["GridPosition"] = pd.to_numeric(
        same_weekend["GridPosition"], errors="coerce"
    )

    same_weekend = same_weekend.drop_duplicates(
        subset=["Year", "Round", "SessionCode", "FullName", "TeamName"],
        keep="last"
    ).copy()

    q_df = same_weekend[same_weekend["SessionCode"] == "Q"][
        ["FullName", "TeamName", "Position"]
    ].drop_duplicates(subset=["FullName", "TeamName"]).rename(
        columns={"Position": "WeekendQPosition"}
    )

    sq_df = same_weekend[same_weekend["SessionCode"] == "SQ"][
        ["FullName", "TeamName", "Position"]
    ].drop_duplicates(subset=["FullName", "TeamName"]).rename(
        columns={"Position": "WeekendSQPosition"}
    )

    s_df = same_weekend[same_weekend["SessionCode"] == "S"][
        ["FullName", "TeamName", "Position"]
    ].drop_duplicates(subset=["FullName", "TeamName"]).rename(
        columns={"Position": "WeekendSPosition"}
    )

    merged = pred_rows.merge(q_df, on=["FullName", "TeamName"], how="left")
    merged = merged.merge(sq_df, on=["FullName", "TeamName"], how="left")
    merged = merged.merge(s_df, on=["FullName", "TeamName"], how="left")

    return merged


def build_current_score(pred_rows: pd.DataFrame, session_code: str) -> pd.Series:
    score = pd.Series(0.0, index=pred_rows.index, dtype=float)

    def fill_series(col_name: str) -> pd.Series:
        if col_name not in pred_rows.columns:
            return pd.Series(0.0, index=pred_rows.index, dtype=float)

        s = pd.to_numeric(pred_rows[col_name], errors="coerce")

        if s.notna().sum() == 0:
            return pd.Series(0.0, index=pred_rows.index, dtype=float)

        median = s.median()
        return s.fillna(median)

    # Practice pace
    score += fill_series("PracticeBestLap") * 0.45
    score += fill_series("PracticeAvgLap") * 0.20

    # Current-season form
    score += fill_series("current_season_driver_avg_pos") * 0.20
    score += fill_series("current_season_team_avg_pos") * 0.15

    # Same-weekend session info
    if session_code == "R":
        score += fill_series("WeekendQPosition") * 0.75
        score += fill_series("WeekendSQPosition") * 0.10
        score += fill_series("WeekendSPosition") * 0.15

    if session_code == "S":
        score += fill_series("WeekendSQPosition") * 0.35

    return score


def predict_session(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    history_file = RAW_DIR / "historical_session_results.csv"
    history_df = pd.read_csv(history_file)

    pred_rows = build_prediction_rows(history_df, year, round_number, session_code)
    pred_rows = attach_practice_features(pred_rows, year, round_number)
    pred_rows = attach_current_weekend_results(pred_rows, year, round_number)

    pred_rows = pred_rows.drop_duplicates(
        subset=["FullName", "TeamName"],
        keep="last"
    ).reset_index(drop=True)

    model_path = f"models/{session_code}_model.pkl"
    bundle = joblib.load(model_path)
    model = bundle["model"]

    X, _, _, _ = encode_features(pred_rows)
    pred_rows["PriorModelScore"] = model.predict(X)
    pred_rows["CurrentSeasonScore"] = build_current_score(pred_rows, session_code)

    if year == CURRENT_SEASON:
        prior_weight = get_prior_weight(round_number)
        current_weight = get_current_weight(round_number)
    else:
        prior_weight = 1.0
        current_weight = 0.0

    pred_rows["FinalScore"] = (
        pred_rows["PriorModelScore"] * prior_weight
        + pred_rows["CurrentSeasonScore"] * current_weight
    )

    pred_rows = pred_rows.sort_values("FinalScore").reset_index(drop=True)
    pred_rows["PredictedRank"] = pred_rows.index + 1
    pred_rows["PriorWeight"] = prior_weight
    pred_rows["CurrentWeight"] = current_weight

    out_file = OUTPUTS_DIR / f"{year}_{round_number}_{session_code}_predictions.csv"
    pred_rows.to_csv(out_file, index=False)

    print(pred_rows[[
        "PredictedRank",
        "FullName",
        "TeamName",
        "PriorModelScore",
        "CurrentSeasonScore",
        "FinalScore"
    ]])
    print(f"Saved predictions to {out_file}")

    return pred_rows


if __name__ == "__main__":
    year = DEFAULT_PREDICT_YEAR
    round_number = DEFAULT_PREDICT_ROUND
    session_code = "Q"

    if len(sys.argv) >= 4:
        year = int(sys.argv[1])
        round_number = int(sys.argv[2])
        session_code = sys.argv[3].upper()

    predict_session(year, round_number, session_code)