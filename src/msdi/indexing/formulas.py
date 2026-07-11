import numpy as np
import pandas as pd


def compute_event_score(df: pd.DataFrame) -> pd.Series:
    """Compute article/event score from relevance, severity, confidence, and novelty.

    Expected columns:
    - relevance_probability
    - severity
    - confidence
    - novelty_weight
    """
    required = ["relevance_probability", "severity", "confidence", "novelty_weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return (
        df["relevance_probability"].astype(float)
        * df["severity"].astype(float)
        * df["confidence"].astype(float)
        * df["novelty_weight"].astype(float)
    )


def aggregate_period_index(
    df: pd.DataFrame,
    date_col: str = "event_date",
    freq: str = "W",
    media_volume_col: str | None = "media_volume",
) -> pd.DataFrame:
    """Aggregate event scores into a media-volume-corrected z-scored index."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if "event_score" not in work.columns:
        work["event_score"] = compute_event_score(work)

    grouped = work.set_index(date_col).resample(freq).agg(
        raw_score=("event_score", "sum"),
        event_count=("event_score", "size"),
    )

    if media_volume_col and media_volume_col in work.columns:
        media = work.set_index(date_col).resample(freq)[media_volume_col].sum()
        grouped["media_volume"] = media
        grouped["volume_corrected_score"] = grouped["raw_score"] / np.log1p(grouped["media_volume"].clip(lower=0))
    else:
        grouped["media_volume"] = np.nan
        grouped["volume_corrected_score"] = grouped["raw_score"]

    s = grouped["volume_corrected_score"]
    grouped["msdi"] = (s - s.mean()) / s.std(ddof=0)
    return grouped.reset_index()
