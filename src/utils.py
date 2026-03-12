from config import EARLY_SEASON_PRIOR_WEIGHTS, DEFAULT_PRIOR_WEIGHT


def get_prior_weight(round_number: int) -> float:
    return EARLY_SEASON_PRIOR_WEIGHTS.get(round_number, DEFAULT_PRIOR_WEIGHT)


def get_current_weight(round_number: int) -> float:
    return 1.0 - get_prior_weight(round_number)


def safe_round_number(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0