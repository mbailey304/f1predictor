import sys
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, DEFAULT_PREDICT_YEAR, DEFAULT_PREDICT_ROUND


def load_race_predictions(year: int, round_number: int) -> pd.DataFrame:
    path = OUTPUTS_DIR / f"{year}_{round_number}_R_predictions.csv"
    df = pd.read_csv(path)

    needed_cols = ["FullName", "TeamName", "PredictedRank", "FinalScore"]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df.copy()


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def build_driver_strength(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Lower FinalScore is better, so invert it into strength
    work["BaseStrength"] = -pd.to_numeric(work["FinalScore"], errors="coerce")

    # Optional boost from grid position if present
    if "WeekendQPosition" in work.columns:
        qpos = pd.to_numeric(work["WeekendQPosition"], errors="coerce")
        work["GridStrength"] = -qpos.fillna(qpos.median())
    else:
        work["GridStrength"] = 0.0

    # Optional practice pace effect
    if "PracticeBestLap" in work.columns:
        pbest = pd.to_numeric(work["PracticeBestLap"], errors="coerce")
        work["PracticeStrength"] = -pbest.fillna(pbest.median())
    else:
        work["PracticeStrength"] = 0.0

    # Normalize each component
    work["BaseStrengthNorm"] = normalize_series(work["BaseStrength"])
    work["GridStrengthNorm"] = normalize_series(work["GridStrength"])
    work["PracticeStrengthNorm"] = normalize_series(work["PracticeStrength"])

    # Combined driver strength
    work["DriverStrength"] = (
        work["BaseStrengthNorm"] * 0.60 +
        work["GridStrengthNorm"] * 0.25 +
        work["PracticeStrengthNorm"] * 0.15
    )

    return work


def assign_dnf_risk(df: pd.DataFrame) -> pd.Series:
    """
    Very simple starter DNF model.
    Later you can replace this with:
    - driver finish rate
    - team finish rate
    - wet race risk
    - first-lap incident risk
    """
    base_risk = 0.08  # 8% starter probability
    return pd.Series(base_risk, index=df.index)


def simulate_once(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sim = df.copy()

    # Baseline strength
    strength = sim["DriverStrength"].to_numpy()

    # Random components
    start_noise = rng.normal(0, 0.35, len(sim))
    strategy_noise = rng.normal(0, 0.45, len(sim))
    race_noise = rng.normal(0, 0.60, len(sim))

    # DNF simulation
    dnf_risk = sim["DNFRisk"].to_numpy()
    dnf_flags = rng.random(len(sim)) < dnf_risk

    # Final simulated score
    # Higher score = better
    sim["SimScore"] = strength + start_noise + strategy_noise + race_noise

    # DNFs get pushed to bottom
    sim.loc[dnf_flags, "SimScore"] = -9999
    sim["DNF"] = dnf_flags

    # Rank finishers first by descending score
    sim = sim.sort_values("SimScore", ascending=False).reset_index(drop=True)

    # Assign positions
    sim["SimFinish"] = range(1, len(sim) + 1)

    # Move DNFs to back while preserving their relative simulated order
    finishers = sim[~sim["DNF"]].copy()
    dnfs = sim[sim["DNF"]].copy()

    sim = pd.concat([finishers, dnfs], ignore_index=True)
    sim["SimFinish"] = range(1, len(sim) + 1)

    return sim[["FullName", "TeamName", "SimFinish", "DNF"]]


def run_monte_carlo(year: int, round_number: int, n_sims: int = 10000, seed: int = 42) -> pd.DataFrame:
    base = load_race_predictions(year, round_number)
    base = build_driver_strength(base)
    base["DNFRisk"] = assign_dnf_risk(base)

    rng = np.random.default_rng(seed)

    all_results = []

    for sim_id in range(n_sims):
        sim_result = simulate_once(base, rng)
        sim_result["Simulation"] = sim_id
        all_results.append(sim_result)

    sims = pd.concat(all_results, ignore_index=True)

    summary = sims.groupby(["FullName", "TeamName"]).agg(
        AvgFinish=("SimFinish", "mean"),
        BestFinish=("SimFinish", "min"),
        WorstFinish=("SimFinish", "max"),
        WinProbability=("SimFinish", lambda s: (s == 1).mean()),
        PodiumProbability=("SimFinish", lambda s: (s <= 3).mean()),
        Top10Probability=("SimFinish", lambda s: (s <= 10).mean()),
        DNFProbability=("DNF", "mean"),
    ).reset_index()

    summary = summary.sort_values(["WinProbability", "PodiumProbability", "AvgFinish"], ascending=[False, False, True])
    out_file = OUTPUTS_DIR / f"{year}_{round_number}_R_simulation_summary.csv"
    summary.to_csv(out_file, index=False)

    sims_file = OUTPUTS_DIR / f"{year}_{round_number}_R_simulation_full.csv"
    sims.to_csv(sims_file, index=False)

    print(summary.head(15))
    print(f"\nSaved summary to {out_file}")
    print(f"Saved full simulations to {sims_file}")

    return summary


if __name__ == "__main__":
    year = DEFAULT_PREDICT_YEAR
    round_number = DEFAULT_PREDICT_ROUND
    n_sims = 10000

    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        round_number = int(sys.argv[2])

    if len(sys.argv) >= 4:
        n_sims = int(sys.argv[3])

    run_monte_carlo(year, round_number, n_sims=n_sims)