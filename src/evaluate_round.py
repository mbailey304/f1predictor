import sys
import numpy as np
import pandas as pd
import fastf1

from config import CACHE_DIR, OUTPUTS_DIR

fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_actual_results(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_code)
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    df = session.results.copy()

    keep_cols = ["FullName", "TeamName", "Position", "GridPosition", "Points", "Status"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    df = df.drop_duplicates(subset=["FullName", "TeamName"], keep="last").reset_index(drop=True)
    return df


def spearman_rank_corr(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return x_rank.corr(y_rank)


def evaluate_session(year: int, round_number: int, session_code: str):
    pred_file = OUTPUTS_DIR / f"{year}_{round_number}_{session_code}_predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    pred = pd.read_csv(pred_file)
    actual = load_actual_results(year, round_number, session_code)

    pred["PredictedRank"] = pd.to_numeric(pred["PredictedRank"], errors="coerce")
    actual["Position"] = pd.to_numeric(actual["Position"], errors="coerce")

    pred = pred.drop_duplicates(subset=["FullName", "TeamName"], keep="last").copy()
    actual = actual.drop_duplicates(subset=["FullName", "TeamName"], keep="last").copy()

    merged = pred.merge(
        actual,
        on=["FullName", "TeamName"],
        how="inner",
        suffixes=("_pred", "_actual")
    )

    merged["AbsError"] = (merged["PredictedRank"] - merged["Position_actual"]).abs()

    mae = merged["AbsError"].mean()
    exact_matches = (merged["PredictedRank"] == merged["Position_actual"]).sum()
    exact_match_rate = exact_matches / len(merged) if len(merged) else np.nan

    top10_pred = set(merged.nsmallest(10, "PredictedRank")["FullName"])
    top10_actual = set(merged.nsmallest(10, "Position_actual")["FullName"])
    top10_overlap = len(top10_pred & top10_actual)

    spearman = spearman_rank_corr(merged["PredictedRank"], merged["Position_actual"])

    print(f"\n=== {session_code} Evaluation: {year} Round {round_number} ===")
    print(f"Rows compared: {len(merged)}")
    print(f"MAE: {mae:.3f}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match rate: {exact_match_rate:.3%}")
    print(f"Top-10 overlap: {top10_overlap}/10")
    print(f"Spearman rank correlation: {spearman:.3f}")

    print("\nWorst misses:")
    worst = merged.sort_values("AbsError", ascending=False).head(10)
    print(worst[[
        "PredictedRank",
        "Position_actual",
        "FullName",
        "TeamName",
        "AbsError"
    ]])

    out_file = OUTPUTS_DIR / f"{year}_{round_number}_{session_code}_evaluation.csv"
    merged.to_csv(out_file, index=False)
    print(f"\nSaved detailed evaluation to {out_file}")


if __name__ == "__main__":
    year = 2026
    round_number = 1
    session_code = "Q"

    if len(sys.argv) >= 4:
        year = int(sys.argv[1])
        round_number = int(sys.argv[2])
        session_code = sys.argv[3].upper()

    evaluate_session(year, round_number, session_code)