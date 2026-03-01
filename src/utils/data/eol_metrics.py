from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_days_to_eol(eol_date_series: pd.Series, today: pd.Timestamp | None = None) -> pd.Series:
    """
    Compute days-to-EoL for each device row.

    Parameters
    ----------
    eol_date_series : pd.Series
        Series containing recommended EoL dates.
    today : pd.Timestamp, optional
        Reference date. If not provided, uses the local current date (normalized).

    Returns
    -------
    pd.Series
        Days until EoL. Negative means overdue. NaN means unknown EoL date.
    """
    reference_day = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    eol_dates = pd.to_datetime(eol_date_series, errors='coerce')
    return (eol_dates.dt.normalize() - reference_day).dt.days


def bucket_days_to_eol(days_to_eol_series: pd.Series, horizon_days: int) -> Dict[str, int]:
    """
    Bucket days-to-EoL into overdue / within horizon / future / unknown.
    """
    if horizon_days < 0:
        raise ValueError('horizon_days must be >= 0.')

    numeric_days = pd.to_numeric(days_to_eol_series, errors='coerce')
    total_devices = int(len(numeric_days))

    overdue_count = int((numeric_days < 0).sum())
    within_horizon_count = int(((numeric_days >= 0) & (numeric_days <= horizon_days)).sum())
    future_count = int((numeric_days > horizon_days).sum())
    unknown_count = int(numeric_days.isna().sum())

    if overdue_count + within_horizon_count + future_count + unknown_count != total_devices:
        raise ValueError('EoL category counts do not sum to total_devices.')

    return {
        'total_devices': total_devices,
        'overdue_count': overdue_count,
        'within_horizon_count': within_horizon_count,
        'future_count': future_count,
        'unknown_count': unknown_count,
        'known_count': total_devices - unknown_count,
    }
