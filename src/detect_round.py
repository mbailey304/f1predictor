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

        # Skip testing if possible
        try:
            if event.is_testing():
                continue
        except Exception:
            pass

        event_date = None

        for col in ["Session5Date", "Session4Date", "Session3Date", "Session2Date", "Session1Date", "EventDate"]:
            if col in event and pd.notna(event[col]):
                event_date = to_naive(event[col])
                if event_date is not None:
                    break

        if event_date is None:
            continue

        candidates.append((round_number, event_date))

    if not candidates:
        return 1

    candidates = sorted(candidates, key=lambda x: x[1])

    # If we're before the first race weekend, use round 1
    if now < candidates[0][1]:
        return candidates[0][0]

    latest_round = candidates[0][0]

    for round_number, event_date in candidates:
        if now >= event_date:
            latest_round = round_number
        else:
            break

    return latest_round


if __name__ == "__main__":
    print(detect_current_round())