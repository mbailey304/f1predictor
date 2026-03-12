from datetime import datetime
import pandas as pd
import fastf1

from config import CACHE_DIR

fastf1.Cache.enable_cache(str(CACHE_DIR))


def to_naive(dt):
    """Convert pandas/python datetimes to timezone-naive datetime."""
    if dt is None:
        return None

    ts = pd.Timestamp(dt)

    # If timezone-aware, convert to UTC then drop tz info
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts.to_pydatetime()


def detect_stage(year: int, round_number: int) -> str:
    event = fastf1.get_event(year, round_number)

    # Naive UTC time for clean comparison
    now = datetime.utcnow()

    session_times = []

    for session_name in [
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Sprint Qualifying",
        "Sprint",
        "Qualifying",
        "Race",
    ]:
        try:
            session = event.get_session(session_name)
            session_date = getattr(session, "date", None)

            if session_date is not None:
                session_date = to_naive(session_date)
                session_times.append((session_name, session_date))
        except Exception:
            continue

    session_times = sorted(session_times, key=lambda x: x[1])

    if not session_times:
        return "Manual Update"

    first_session_time = session_times[0][1]
    if now < first_session_time:
        return "Pre-weekend"

    latest_finished = None
    for name, session_time in session_times:
        if now >= session_time:
            latest_finished = name
        else:
            break

    stage_map = {
        "Practice 1": "Post-FP1",
        "Practice 2": "Post-FP2",
        "Practice 3": "Post-FP3",
        "Sprint Qualifying": "Post-Sprint-Qualifying",
        "Sprint": "Post-Sprint",
        "Qualifying": "Post-Qualifying",
        "Race": "Post-Race",
    }

    return stage_map.get(latest_finished, "Manual Update")