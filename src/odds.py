import pandas as pd


def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def decimal_to_implied_prob(decimal_odds):
    decimal_odds = float(decimal_odds)
    if decimal_odds <= 0:
        return None
    return 1.0 / decimal_odds


def expected_value(prob, decimal_odds):
    prob = float(prob)
    decimal_odds = float(decimal_odds)
    return (prob * (decimal_odds - 1)) - (1 - prob)


def kelly_fraction(prob, decimal_odds):
    prob = float(prob)
    decimal_odds = float(decimal_odds)
    b = decimal_odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    k = ((b * prob) - q) / b
    return max(0.0, k)


def normalize_driver_name(name: str) -> str:
    return str(name).strip().lower()


def prepare_odds_table(odds_df: pd.DataFrame) -> pd.DataFrame:
    work = odds_df.copy()

    if "Driver" not in work.columns:
        raise ValueError("Odds data must contain a 'Driver' column")
    if "Market" not in work.columns:
        raise ValueError("Odds data must contain a 'Market' column")
    if "AmericanOdds" not in work.columns:
        raise ValueError("Odds data must contain an 'AmericanOdds' column")

    if "Sportsbook" not in work.columns:
        work["Sportsbook"] = "Manual"

    work["DriverKey"] = work["Driver"].apply(normalize_driver_name)
    work["DecimalOdds"] = work["AmericanOdds"].apply(american_to_decimal)
    work["ImpliedProbability"] = work["DecimalOdds"].apply(decimal_to_implied_prob)

    return work


def build_value_bets(sim_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    sim = sim_df.copy()
    odds = prepare_odds_table(odds_df)

    sim["DriverKey"] = sim["FullName"].apply(normalize_driver_name)

    market_map = {
        "win": "WinProbability",
        "podium": "PodiumProbability",
        "top10": "Top10Probability",
    }

    rows = []

    for _, odd in odds.iterrows():
        market = str(odd["Market"]).strip().lower()
        if market not in market_map:
            continue

        prob_col = market_map[market]
        match = sim[sim["DriverKey"] == odd["DriverKey"]]

        if match.empty or prob_col not in match.columns:
            continue

        model_prob = float(match.iloc[0][prob_col])
        implied_prob = float(odd["ImpliedProbability"])
        decimal_odds = float(odd["DecimalOdds"])

        edge = model_prob - implied_prob
        ev = expected_value(model_prob, decimal_odds)
        kelly = kelly_fraction(model_prob, decimal_odds)

        rows.append({
            "Driver": match.iloc[0]["FullName"],
            "TeamName": match.iloc[0]["TeamName"],
            "Market": market,
            "Sportsbook": odd["Sportsbook"],
            "AmericanOdds": odd["AmericanOdds"],
            "DecimalOdds": round(decimal_odds, 3),
            "ImpliedProbability": implied_prob,
            "ModelProbability": model_prob,
            "Edge": edge,
            "ExpectedValue": ev,
            "KellyFull": kelly,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(["ExpectedValue", "Edge"], ascending=False).reset_index(drop=True)
