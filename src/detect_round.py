from datetime import datetime, timedelta
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

    events = []

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

        session_dates = []

        for col in ["Session1Date", "Session2Date", "Session3Date", "Session4Date", "Session5Date", "EventDate"]:
            if col in event and pd.notna(event[col]):
                dt = to_naive(event[col])
                if dt is not None:
                    session_dates.append(dt)

        if not session_dates:
            continue

        session_dates = sorted(session_dates)

        # Start slightly before the first listed session
        weekend_start = session_dates[0] - timedelta(hours=12)

        # End well after the final listed session to avoid early rollover
        weekend_end = session_dates[-1] + timedelta(hours=36)

        events.append((round_number, weekend_start, weekend_end))

    if not events:
        return 1

    events = sorted(events, key=lambda x: x[1])

    # Before first weekend
    if now < events[0][1]:
        return events[0][0]

    # During an active weekend window
    for round_number, weekend_start, weekend_end in events:
        if weekend_start <= now <= weekend_end:
            return round_number

    # Between weekends -> next upcoming
    for round_number, weekend_start, weekend_end in events:
        if now < weekend_start:
            return round_number

    # After final weekend
    return events[-1][0]


if __name__ == "__main__":
    print(detect_current_round())