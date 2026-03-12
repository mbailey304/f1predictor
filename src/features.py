import numpy as np
import pandas as pd


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["Position", "GridPosition", "Points", "Laps", "Round", "Year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    time_cols = ["Q1", "Q2", "Q3", "Time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()

    sort_cols = [c for c in ["SessionCode", "FullName", "Year", "Round"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "driver_prev_pos_avg_3",
        "driver_prev_pos_avg_5",
        "team_prev_pos_avg_3",
        "driver_prev_points_avg_3",
        "driver_prev_grid_avg_3",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    if {"SessionCode", "FullName", "Position"}.issubset(df.columns):
        df["driver_prev_pos_avg_3"] = (
            df.groupby(["SessionCode", "FullName"])["Position"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

        df["driver_prev_pos_avg_5"] = (
            df.groupby(["SessionCode", "FullName"])["Position"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    if {"SessionCode", "TeamName", "Position"}.issubset(df.columns):
        df["team_prev_pos_avg_3"] = (
            df.groupby(["SessionCode", "TeamName"])["Position"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    if {"SessionCode", "FullName", "Points"}.issubset(df.columns):
        df["driver_prev_points_avg_3"] = (
            df.groupby(["SessionCode", "FullName"])["Points"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    if {"SessionCode", "FullName", "GridPosition"}.issubset(df.columns):
        df["driver_prev_grid_avg_3"] = (
            df.groupby(["SessionCode", "FullName"])["GridPosition"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    return df


def add_current_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "current_season_driver_avg_pos",
        "current_season_team_avg_pos",
        "current_season_driver_avg_grid",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    if {"Year", "SessionCode", "FullName", "Position"}.issubset(df.columns):
        df["current_season_driver_avg_pos"] = (
            df.groupby(["Year", "SessionCode", "FullName"])["Position"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )

    if {"Year", "SessionCode", "TeamName", "Position"}.issubset(df.columns):
        df["current_season_team_avg_pos"] = (
            df.groupby(["Year", "SessionCode", "TeamName"])["Position"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )

    if {"Year", "SessionCode", "FullName", "GridPosition"}.issubset(df.columns):
        df["current_season_driver_avg_grid"] = (
            df.groupby(["Year", "SessionCode", "FullName"])["GridPosition"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )

    return df


def encode_features(df: pd.DataFrame):
    work = df.copy()

    if "FullName" not in work.columns:
        work["FullName"] = "Unknown Driver"
    if "TeamName" not in work.columns:
        work["TeamName"] = "Unknown Team"
    if "EventName" not in work.columns:
        work["EventName"] = "Unknown Event"
    if "SessionCode" not in work.columns:
        work["SessionCode"] = "UNK"

    work["DriverCat"] = work["FullName"].astype("category")
    work["TeamCat"] = work["TeamName"].astype("category")
    work["EventCat"] = work["EventName"].astype("category")
    work["SessionCat"] = work["SessionCode"].astype("category")

    feature_cols = [
        "DriverCat",
        "TeamCat",
        "EventCat",
        "SessionCat",
        "driver_prev_pos_avg_3",
        "driver_prev_pos_avg_5",
        "team_prev_pos_avg_3",
        "driver_prev_points_avg_3",
        "driver_prev_grid_avg_3",
        "current_season_driver_avg_pos",
        "current_season_team_avg_pos",
        "current_season_driver_avg_grid",
    ]

    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan

    X = work[feature_cols].copy()

    for col in ["DriverCat", "TeamCat", "EventCat", "SessionCat"]:
        X[col] = X[col].cat.codes

    X = X.replace(-1, np.nan)

    y = pd.to_numeric(work.get("Position"), errors="coerce")
    groups = (
        work["Year"].astype("Int64").astype(str) + "_"
        + work["Round"].astype("Int64").astype(str)
    )

    return X, y, groups, feature_cols