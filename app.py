import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import KELLY_FRACTION, MIN_EDGE, MIN_EV
from src.odds import build_value_bets

st.set_page_config(page_title="F1 Weekend Predictor", layout="wide")

OUTPUTS_DIR = Path("outputs")
HISTORY_DIR = OUTPUTS_DIR / "history"

TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#FF87BC",
    "Williams": "#005AFF",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
    "RB": "#6692FF",
    "AlphaTauri": "#4E7C9B",
    "Sauber": "#52E252",
    "Audi": "#00A19B",
    "Cadillac": "#003DA5",
    "Kick Sauber": "#52E252",
}

st.markdown(
    """
    <style>
    table { font-size: 15px; width: 100%; border-collapse: collapse; }
    th { text-align: left !important; padding: 8px; }
    td { padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def load_metadata(path: Path):
    return json.loads(path.read_text())


def team_dot_html(team: str) -> str:
    color = TEAM_COLORS.get(team, "#888888")
    return f'<span style="color:{color}; font-size:18px;">●</span> {team}'


def add_team_color_dot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "TeamName" in work.columns:
        work["Team"] = work["TeamName"].astype(str).apply(team_dot_html)
    return work


def render_html_table(df: pd.DataFrame):
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    rename_map = {
        "PredictedRank": "Rank",
        "FullName": "Driver",
        "FinalScore": "Score",
        "PriorWeight": "Prior Wt",
        "CurrentWeight": "Current Wt",
        "PriorModelScore": "Prior Score",
        "CurrentSeasonScore": "Current Score",
        "WeekendQPosition": "Grid",
        "WeekendSQPosition": "SQ Pos",
        "WeekendSPosition": "Sprint Pos",
    }
    work = work.rename(columns=rename_map)

    for col in ["Score", "Prior Score", "Current Score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").round(3)

    for col in ["Prior Wt", "Current Wt"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").round(2)

    return work


def format_sim_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    rename_map = {
        "FullName": "Driver",
        "AvgFinish": "Avg Finish",
        "WinProbability": "Win %",
        "PodiumProbability": "Podium %",
        "Top10Probability": "Top 10 %",
        "DNFProbability": "DNF %",
        "BestFinish": "Best",
        "WorstFinish": "Worst",
    }
    work = work.rename(columns=rename_map)

    if "Avg Finish" in work.columns:
        work["Avg Finish"] = pd.to_numeric(work["Avg Finish"], errors="coerce").round(2)

    for col in ["Win %", "Podium %", "Top 10 %", "DNF %"]:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(work[col], errors="coerce")
                .mul(100)
                .round(1)
                .astype(str)
                + "%"
            )

    return work


def render_top3_cards(df: pd.DataFrame, title: str):
    if not {"PredictedRank", "FullName", "TeamName"}.issubset(df.columns):
        return

    top3 = df.nsmallest(3, "PredictedRank")[["PredictedRank", "FullName", "TeamName"]].copy()

    st.markdown(f"### {title}")
    cols = st.columns(3)

    for i, (_, row) in enumerate(top3.iterrows()):
        team = row["TeamName"]
        color = TEAM_COLORS.get(team, "#666666")
        with cols[i]:
            st.markdown(
                f"""
<div style="
    border-radius: 14px;
    padding: 16px;
    border-left: 8px solid {color};
    background-color: rgba(255,255,255,0.03);
">
    <div style="font-size: 14px; opacity: 0.8;">P{int(row["PredictedRank"])}</div>
    <div style="font-size: 22px; font-weight: 700;">{row["FullName"]}</div>
    <div style="font-size: 15px; opacity: 0.85;">{team}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def show_prediction_table(file_path: Path, title: str):
    st.subheader(title)

    if not file_path.exists():
        st.warning(f"No {title.lower()} file found.")
        return

    df = load_csv(file_path)

    show_cols = [
        c for c in [
            "PredictedRank",
            "FullName",
            "TeamName",
            "FinalScore",
            "PriorWeight",
            "CurrentWeight",
            "PriorModelScore",
            "CurrentSeasonScore",
            "WeekendQPosition",
            "WeekendSQPosition",
            "WeekendSPosition",
        ]
        if c in df.columns
    ]

    display_df = df[show_cols].copy()
    display_df = add_team_color_dot(display_df)
    display_df = format_prediction_table(display_df)

    ordered_cols = [
        c for c in [
            "Rank",
            "Driver",
            "Team",
            "Score",
            "Prior Wt",
            "Current Wt",
            "Prior Score",
            "Current Score",
            "Grid",
            "SQ Pos",
            "Sprint Pos",
        ]
        if c in display_df.columns
    ]

    render_html_table(display_df[ordered_cols])
    render_top3_cards(df, "Projected Top 3")


def show_sim_table(file_path: Path):
    st.subheader("Race Simulation Summary")

    if not file_path.exists():
        st.warning("No race simulation summary found.")
        return

    sim_df = load_csv(file_path)

    sim_cols = [
        c for c in [
            "FullName",
            "TeamName",
            "AvgFinish",
            "WinProbability",
            "PodiumProbability",
            "Top10Probability",
            "DNFProbability",
            "BestFinish",
            "WorstFinish",
        ]
        if c in sim_df.columns
    ]

    display_df = sim_df[sim_cols].copy()
    display_df = add_team_color_dot(display_df)
    display_df = format_sim_table(display_df)

    ordered_cols = [
        c for c in [
            "Driver",
            "Team",
            "Avg Finish",
            "Win %",
            "Podium %",
            "Top 10 %",
            "DNF %",
            "Best",
            "Worst",
        ]
        if c in display_df.columns
    ]

    render_html_table(display_df[ordered_cols])

    if {"FullName", "TeamName", "WinProbability", "PodiumProbability", "AvgFinish"}.issubset(sim_df.columns):
        st.markdown("### Most likely winner candidates")
        top_win = sim_df.sort_values("WinProbability", ascending=False).head(5).copy()
        top_win = add_team_color_dot(top_win)
        top_win = format_sim_table(top_win)

        cols = [c for c in ["Driver", "Team", "Win %", "Podium %", "Avg Finish"] if c in top_win.columns]
        render_html_table(top_win[cols])


def build_prediction_shift_table(q_file: Path, r_file: Path):
    if not q_file.exists() or not r_file.exists():
        return None

    q_df = load_csv(q_file)[["FullName", "TeamName", "PredictedRank"]].copy()
    r_df = load_csv(r_file)[["FullName", "TeamName", "PredictedRank"]].copy()

    q_df = q_df.rename(columns={"PredictedRank": "Q Pred"})
    r_df = r_df.rename(columns={"PredictedRank": "Race Pred"})

    merged = q_df.merge(r_df, on=["FullName", "TeamName"], how="inner")
    merged["Q Pred"] = pd.to_numeric(merged["Q Pred"], errors="coerce")
    merged["Race Pred"] = pd.to_numeric(merged["Race Pred"], errors="coerce")
    merged["DeltaValue"] = merged["Q Pred"] - merged["Race Pred"]

    def delta_display(val):
        if pd.isna(val):
            return "—"
        val = int(val)
        if val > 0:
            return f"↑ {val}"
        if val < 0:
            return f"↓ {abs(val)}"
        return "→ 0"

    merged["Δ"] = merged["DeltaValue"].apply(delta_display)
    merged = merged.sort_values(["DeltaValue", "Race Pred"], ascending=[False, True]).reset_index(drop=True)
    merged = add_team_color_dot(merged)
    merged = merged.rename(columns={"FullName": "Driver"})

    return merged


def show_prediction_shift_table(q_file: Path, r_file: Path):
    st.subheader("Weekend Prediction Shift")
    st.caption("Compares predicted qualifying rank vs predicted race rank.")

    shift_df = build_prediction_shift_table(q_file, r_file)
    if shift_df is None or shift_df.empty:
        st.warning("Need both qualifying and race predictions.")
        return

    display_cols = ["Driver", "Team", "Q Pred", "Race Pred", "Δ"]
    render_html_table(shift_df[display_cols])


def parse_snapshot_info(path: Path):
    stem = path.stem
    parts = stem.split("__")

    if len(parts) >= 3:
        base_name = parts[0]
        stage = parts[1]
        timestamp = parts[2]
    else:
        base_name = stem
        stage = "Unknown"
        timestamp = "Unknown"

    return {
        "path": path,
        "base_name": base_name,
        "stage": stage.replace("-", " "),
        "timestamp": timestamp,
    }


def get_snapshot_files(year: int, round_number: int, session_code: str):
    if not HISTORY_DIR.exists():
        return []

    pattern = f"{year}_{round_number}_{session_code}_predictions__*__*.csv"
    files = sorted(HISTORY_DIR.glob(pattern))
    return [parse_snapshot_info(f) for f in files]


def build_update_shift_table(year: int, round_number: int, session_code: str):
    files = get_snapshot_files(year, round_number, session_code)
    if len(files) < 2:
        return None

    previous_info = files[-2]
    latest_info = files[-1]

    prev_df = load_csv(previous_info["path"])[["FullName", "TeamName", "PredictedRank"]].copy()
    curr_df = load_csv(latest_info["path"])[["FullName", "TeamName", "PredictedRank"]].copy()

    prev_df = prev_df.rename(columns={"PredictedRank": "Previous"})
    curr_df = curr_df.rename(columns={"PredictedRank": "Latest"})

    merged = prev_df.merge(curr_df, on=["FullName", "TeamName"], how="inner")
    merged["Previous"] = pd.to_numeric(merged["Previous"], errors="coerce")
    merged["Latest"] = pd.to_numeric(merged["Latest"], errors="coerce")
    merged["DeltaValue"] = merged["Previous"] - merged["Latest"]

    def delta_display(val):
        if pd.isna(val):
            return "—"
        val = int(val)
        if val > 0:
            return f"↑ {val}"
        if val < 0:
            return f"↓ {abs(val)}"
        return "→ 0"

    merged["Δ"] = merged["DeltaValue"].apply(delta_display)
    merged = merged.sort_values(["DeltaValue", "Latest"], ascending=[False, True]).reset_index(drop=True)
    merged = add_team_color_dot(merged)
    merged = merged.rename(columns={"FullName": "Driver"})

    return merged, previous_info, latest_info


def show_update_shift_table(year: int, round_number: int, session_code: str, label: str):
    st.subheader(f"{label} Update Tracker")
    st.caption("Shows how predictions changed from the previous saved stage to the latest one.")

    result = build_update_shift_table(year, round_number, session_code)
    if result is None:
        st.warning("Need at least two saved snapshots to show update movement.")
        return

    shift_df, prev_info, latest_info = result

    st.caption(
        f"Comparing: **{prev_info['stage']}** ({prev_info['timestamp']}) "
        f"→ **{latest_info['stage']}** ({latest_info['timestamp']})"
    )

    display_cols = ["Driver", "Team", "Previous", "Latest", "Δ"]
    render_html_table(shift_df[display_cols])

    gainers = shift_df[shift_df["DeltaValue"] > 0].head(5)
    fallers = shift_df[shift_df["DeltaValue"] < 0].sort_values("DeltaValue").head(5)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Biggest movers up")
        if not gainers.empty:
            render_html_table(gainers[display_cols])
        else:
            st.info("No positive movers yet.")

    with c2:
        st.markdown("### Biggest movers down")
        if not fallers.empty:
            render_html_table(fallers[display_cols])
        else:
            st.info("No negative movers yet.")


def show_probability_charts(file_path: Path):
    st.subheader("Race Probability Charts")

    if not file_path.exists():
        st.warning("No simulation summary found.")
        return

    sim_df = load_csv(file_path).copy()

    needed = ["FullName", "TeamName", "WinProbability", "PodiumProbability", "Top10Probability"]
    available = [c for c in needed if c in sim_df.columns]
    sim_df = sim_df[available].copy()

    for col in ["WinProbability", "PodiumProbability", "Top10Probability"]:
        if col in sim_df.columns:
            sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce")

    if "WinProbability" in sim_df.columns:
        st.markdown("### Win Probability")
        win_df = sim_df.sort_values("WinProbability", ascending=False).head(10)

        win_chart = (
            alt.Chart(win_df)
            .mark_bar()
            .encode(
                x=alt.X("WinProbability:Q", title="Win Probability", axis=alt.Axis(format="%")),
                y=alt.Y("FullName:N", sort="-x", title="Driver"),
                tooltip=[
                    alt.Tooltip("FullName:N", title="Driver"),
                    alt.Tooltip("TeamName:N", title="Team"),
                    alt.Tooltip("WinProbability:Q", title="Win %", format=".1%"),
                ],
            )
        )
        st.altair_chart(win_chart, use_container_width=True)

    if "PodiumProbability" in sim_df.columns:
        st.markdown("### Podium Probability")
        podium_df = sim_df.sort_values("PodiumProbability", ascending=False).head(10)

        podium_chart = (
            alt.Chart(podium_df)
            .mark_bar()
            .encode(
                x=alt.X("PodiumProbability:Q", title="Podium Probability", axis=alt.Axis(format="%")),
                y=alt.Y("FullName:N", sort="-x", title="Driver"),
                tooltip=[
                    alt.Tooltip("FullName:N", title="Driver"),
                    alt.Tooltip("TeamName:N", title="Team"),
                    alt.Tooltip("PodiumProbability:Q", title="Podium %", format=".1%"),
                ],
            )
        )
        st.altair_chart(podium_chart, use_container_width=True)

    if "Top10Probability" in sim_df.columns:
        st.markdown("### Top 10 Probability")
        top10_df = sim_df.sort_values("Top10Probability", ascending=False).head(10)

        top10_chart = (
            alt.Chart(top10_df)
            .mark_bar()
            .encode(
                x=alt.X("Top10Probability:Q", title="Top 10 Probability", axis=alt.Axis(format="%")),
                y=alt.Y("FullName:N", sort="-x", title="Driver"),
                tooltip=[
                    alt.Tooltip("FullName:N", title="Driver"),
                    alt.Tooltip("TeamName:N", title="Team"),
                    alt.Tooltip("Top10Probability:Q", title="Top 10 %", format=".1%"),
                ],
            )
        )
        st.altair_chart(top10_chart, use_container_width=True)


def show_manual_value_bets(sim_file: Path):
    st.subheader("Manual Odds Value Bets")
    st.caption("Enter sportsbook odds for win, podium, and top 10 markets, then compare them to your model probabilities.")

    if not sim_file.exists():
        st.warning("No race simulation summary found.")
        return

    sim_df = load_csv(sim_file).copy()

    required_cols = ["FullName", "TeamName", "WinProbability", "PodiumProbability", "Top10Probability"]
    if not all(col in sim_df.columns for col in required_cols):
        st.warning("Simulation summary is missing required columns.")
        return

    driver_rows = sim_df[["FullName", "TeamName"]].copy().drop_duplicates()
    driver_rows = driver_rows.sort_values("FullName").reset_index(drop=True)

    input_rows = []
    for _, row in driver_rows.iterrows():
        input_rows.append({
            "Driver": row["FullName"],
            "Market": "win",
            "Sportsbook": "Manual",
            "AmericanOdds": None,
        })
        input_rows.append({
            "Driver": row["FullName"],
            "Market": "podium",
            "Sportsbook": "Manual",
            "AmericanOdds": None,
        })
        input_rows.append({
            "Driver": row["FullName"],
            "Market": "top10",
            "Sportsbook": "Manual",
            "AmericanOdds": None,
        })

    odds_input_df = pd.DataFrame(input_rows)

    st.markdown("### Enter Odds")
    edited_df = st.data_editor(
        odds_input_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Driver": st.column_config.TextColumn(disabled=True),
            "Market": st.column_config.TextColumn(disabled=True),
            "Sportsbook": st.column_config.TextColumn(help="Enter book name if you want"),
            "AmericanOdds": st.column_config.NumberColumn(
                help="Examples: +350 as 350, -120 as -120",
                step=1,
            ),
        },
        key="manual_odds_editor",
    )

    valid_odds = edited_df.dropna(subset=["AmericanOdds"]).copy()

    if valid_odds.empty:
        st.info("Enter at least one odds value to calculate edges.")
        return

    try:
        value_df = build_value_bets(sim_df, valid_odds)
    except Exception as e:
        st.error(f"Could not calculate value bets: {e}")
        return

    if value_df.empty:
        st.warning("No valid comparisons were produced.")
        return

    filtered = value_df[
        (value_df["Edge"] >= MIN_EDGE) &
        (value_df["ExpectedValue"] >= MIN_EV)
    ].copy()

    all_display = value_df.copy()
    all_display["ImpliedProbability"] = (all_display["ImpliedProbability"] * 100).round(1).astype(str) + "%"
    all_display["ModelProbability"] = (all_display["ModelProbability"] * 100).round(1).astype(str) + "%"
    all_display["Edge"] = (pd.to_numeric(all_display["Edge"]) * 100).round(1).astype(str) + "%"
    all_display["ExpectedValue"] = (pd.to_numeric(all_display["ExpectedValue"]) * 100).round(1).astype(str) + "%"
    all_display["SuggestedStake"] = (
        (pd.to_numeric(all_display["KellyFull"]) * KELLY_FRACTION * 100)
        .round(2)
        .astype(str) + "% bankroll"
    )

    filtered_display = filtered.copy()
    if not filtered_display.empty:
        filtered_display["ImpliedProbability"] = (filtered_display["ImpliedProbability"] * 100).round(1).astype(str) + "%"
        filtered_display["ModelProbability"] = (filtered_display["ModelProbability"] * 100).round(1).astype(str) + "%"
        filtered_display["Edge"] = (pd.to_numeric(filtered_display["Edge"]) * 100).round(1).astype(str) + "%"
        filtered_display["ExpectedValue"] = (pd.to_numeric(filtered_display["ExpectedValue"]) * 100).round(1).astype(str) + "%"
        filtered_display["SuggestedStake"] = (
            (pd.to_numeric(filtered_display["KellyFull"]) * KELLY_FRACTION * 100)
            .round(2)
            .astype(str) + "% bankroll"
        )

    st.markdown("### Best Value Bets")
    if filtered_display.empty:
        st.info("No bets currently clear your edge and EV thresholds.")
    else:
        st.dataframe(
            filtered_display[
                [
                    "Driver",
                    "TeamName",
                    "Market",
                    "Sportsbook",
                    "AmericanOdds",
                    "ImpliedProbability",
                    "ModelProbability",
                    "Edge",
                    "ExpectedValue",
                    "SuggestedStake",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### All Entered Odds vs Model")
    st.dataframe(
        all_display[
            [
                "Driver",
                "TeamName",
                "Market",
                "Sportsbook",
                "AmericanOdds",
                "ImpliedProbability",
                "ModelProbability",
                "Edge",
                "ExpectedValue",
                "SuggestedStake",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


st.title("🏁 F1 Weekend Predictor")

metadata_file = OUTPUTS_DIR / "metadata.json"
if metadata_file.exists():
    meta = load_metadata(metadata_file)
    year = meta.get("year", 2026)
    round_number = meta.get("round", 1)
    event_name = meta.get("event_name", f"Round {round_number}")
    last_updated = meta.get("last_updated", "Unknown")
    current_stage = meta.get("stage", "Unknown")
else:
    year = 2026
    round_number = 1
    event_name = f"Round {round_number}"
    last_updated = "Unknown"
    current_stage = "Unknown"

st.markdown(
    f"""
<div style="padding: 6px 0 14px 0;">
    <div style="font-size: 30px; font-weight: 700;">{event_name}</div>
    <div style="font-size: 16px; opacity: 0.8;">Season {year} • Round {round_number}</div>
</div>
""",
    unsafe_allow_html=True
)

m1, m2 = st.columns(2)
m1.metric("Stage", current_stage)
m2.metric("Last Updated", last_updated)

q_file = OUTPUTS_DIR / f"{year}_{round_number}_Q_predictions.csv"
r_file = OUTPUTS_DIR / f"{year}_{round_number}_R_predictions.csv"
sq_file = OUTPUTS_DIR / f"{year}_{round_number}_SQ_predictions.csv"
s_file = OUTPUTS_DIR / f"{year}_{round_number}_S_predictions.csv"
sim_file = OUTPUTS_DIR / f"{year}_{round_number}_R_simulation_summary.csv"

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🏎️ Qualifying",
        "🏁 Race",
        "📈 Weekend Shift",
        "🕒 Update Tracker",
        "⚡ Sprint Qualifying",
        "⚡ Sprint",
        "📊 Race Probabilities",
        "💰 Value Bets",
    ]
)

with tab1:
    show_prediction_table(q_file, "Predicted Qualifying Order")

with tab2:
    show_prediction_table(r_file, "Predicted Race Order")

with tab3:
    show_prediction_shift_table(q_file, r_file)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        show_update_shift_table(year, round_number, "Q", "Qualifying")
    with c2:
        show_update_shift_table(year, round_number, "R", "Race")

with tab5:
    show_prediction_table(sq_file, "Predicted Sprint Qualifying Order")

with tab6:
    show_prediction_table(s_file, "Predicted Sprint Order")

with tab7:
    show_sim_table(sim_file)
    st.markdown("---")
    show_probability_charts(sim_file)

with tab8:
    show_manual_value_bets(sim_file)