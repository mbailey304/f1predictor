from datetime import datetime
import pandas as pd
import fastf1

from config import CACHE_DIR, CURRENT_SEASON

fastf1.Cache.enable_cache(str(CACHE_DIR))


def to_naive(dt):
    if dt is None:
        return None

    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts.to_pydatetime()


def detect_current_round(year: int = CURRENT_SEASON) -> int:
    schedule = fastf1.get_event_schedule(year)
    now = datetime.utcnow()

    candidates = []

    for _, event in schedule.iterrows():
        try:
            round_number = int(event["RoundNumber"])
        except Exception:
            continue

        try:
            if event.is_testing():
                continue
        except Exception:
            pass

        first_session_time = None

        for col in ["Session1Date", "Session2Date", "Session3Date", "Session4Date", "Session5Date", "EventDate"]:
            if col in event and pd.notna(event[col]):
                first_session_time = to_naive(event[col])
                if first_session_time is not None:
                    break

        if first_session_time is None:
            continue

        candidates.append((round_number, first_session_time))

    if not candidates:
        return 1

    candidates = sorted(candidates, key=lambda x: x[1])

    # If we're before the first event, use round 1
    if now < candidates[0][1]:
        return candidates[0][0]

    # Return the first round whose first session has not happened yet
    for round_number, first_session_time in candidates:
        if now < first_session_time:
            return round_number

    # If all weekends have started already, return the last one
    return candidates[-1][0]


if __name__ == "__main__":
    print(detect_current_round())