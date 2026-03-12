import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from config import RAW_DIR, MODELS_DIR
from features import clean_results, add_form_features, add_current_season_features, encode_features


def train_one_session(df: pd.DataFrame, session_code: str):
    data = df[df["SessionCode"] == session_code].copy()
    data = data.dropna(subset=["Position"])

    if data.empty:
        print(f"No data for {session_code}")
        return

    data = clean_results(data)
    data = add_form_features(data)
    data = add_current_season_features(data)

    X, y, groups, feature_cols = encode_features(data)

    valid_mask = ~y.isna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    groups = groups.loc[valid_mask].reset_index(drop=True)

    if len(X) < 20:
        print(f"Not enough usable rows for {session_code}")
        return

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "session_code": session_code,
        "mae": float(mae),
    }

    out_path = MODELS_DIR / f"{session_code}_model.pkl"
    joblib.dump(bundle, out_path)

    print(f"{session_code} model saved to {out_path}")
    print(f"{session_code} MAE: {mae:.3f}")


if __name__ == "__main__":
    infile = RAW_DIR / "historical_session_results.csv"
    df = pd.read_csv(infile)

    for code in ["Q", "R", "SQ", "S"]:
        train_one_session(df, code)