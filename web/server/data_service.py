from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.data import apply_filter, bucket_days_to_eol, compute_days_to_eol, get_device_dataset, wrangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DEVICE_DATASET_CSV = DATA_DIR / 'device_dataset.csv'
RAW_WORKBOOK = DATA_DIR / 'UAInnovateDataset-SoCo.xlsx'
UNKNOWN_EOL_PREDICTIONS_CSV = DATA_DIR / 'unknown_eol_device_behavior_predictions.csv'
DEVICE_BEHAVIOR_PREDICTIONS_CSV = DATA_DIR / 'device_behavior_predictions.csv'
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODEL_PERFORMANCE_PNG = REPORTS_DIR / 'model_performance.png'
LOGS_DIR = PROJECT_ROOT / 'logs'
RANDOM_FOREST_MODEL_PKL = LOGS_DIR / 'random_forest_model.pkl'


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.upper().replace('', pd.NA)


@lru_cache(maxsize=1)
def load_device_dataset() -> pd.DataFrame:
    if DEVICE_DATASET_CSV.exists():
        device_df = pd.read_csv(DEVICE_DATASET_CSV, low_memory=False)
    elif RAW_WORKBOOK.exists():
        sheets = apply_filter(wrangle(RAW_WORKBOOK))
        device_df = get_device_dataset(sheets)
    else:
        raise FileNotFoundError(
            f'No device dataset found at {DEVICE_DATASET_CSV} and no workbook at {RAW_WORKBOOK}.'
        )

    required_columns = {
        'device_name',
        'loc_state',
        'loc_site_code',
        'loc_site_name',
        'loc_latitude',
        'loc_longitude',
        'modeldata_eol',
        'device_source',
    }
    missing = sorted(required_columns - set(device_df.columns))
    if missing:
        raise ValueError(f'Device dataset is missing required columns: {missing}')

    frame = device_df.copy()
    frame['loc_state'] = _normalize_text(frame['loc_state'])
    frame['loc_site_code'] = _normalize_text(frame['loc_site_code'])
    frame['loc_site_name'] = frame['loc_site_name'].astype('string').str.strip()
    frame['loc_latitude'] = pd.to_numeric(frame['loc_latitude'], errors='coerce')
    frame['loc_longitude'] = pd.to_numeric(frame['loc_longitude'], errors='coerce')
    frame['modeldata_eol'] = pd.to_datetime(frame['modeldata_eol'], errors='coerce')
    frame['device_source'] = frame['device_source'].astype('string').str.strip()
    frame['device_source'] = frame['device_source'].replace({'': pd.NA, 'NAN': pd.NA})
    frame['device_source'] = frame['device_source'].fillna('NA')

    return frame


def _prepare_cluster_frame() -> pd.DataFrame:
    frame = _prepare_mappable_device_frame()

    # Use normalized local date for EoL deltas.
    frame['remaining_days'] = compute_days_to_eol(frame['modeldata_eol'])
    frame['is_overdue'] = frame['remaining_days'] < 0
    # Overdue devices should pull the color further toward red.
    frame['remaining_days_weighted'] = frame['remaining_days'].where(
        frame['remaining_days'] >= 0,
        frame['remaining_days'] * 2,
    )

    clusters = (
        frame.groupby(
            ['loc_state', 'loc_site_code', 'loc_site_name', 'loc_latitude', 'loc_longitude'],
            dropna=False,
            as_index=False,
        )
        .agg(
            device_count=('device_name', 'size'),
            eol_known_devices=('remaining_days', 'count'),
            overdue_devices=('is_overdue', 'sum'),
            remaining_days_total=('remaining_days', lambda values: values.sum(min_count=1)),
            remaining_days_avg=('remaining_days', 'mean'),
            remaining_days_weighted_total=('remaining_days_weighted', lambda values: values.sum(min_count=1)),
        )
        .sort_values(['device_count', 'loc_state', 'loc_site_code'], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    clusters['cluster_id'] = clusters['loc_state'] + '-' + clusters['loc_site_code']
    return clusters


def _prepare_mappable_device_frame() -> pd.DataFrame:
    frame = load_device_dataset().copy()
    return frame.dropna(subset=['loc_state', 'loc_site_code', 'loc_latitude', 'loc_longitude']).copy()


def _safe_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


def _safe_str(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    return str(value)


def _safe_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


@lru_cache(maxsize=1)
def load_inference_predictions() -> pd.DataFrame:
    candidate_paths = [UNKNOWN_EOL_PREDICTIONS_CSV, DEVICE_BEHAVIOR_PREDICTIONS_CSV]
    for csv_path in candidate_paths:
        if csv_path.exists():
            frame = pd.read_csv(csv_path, low_memory=False)
            break
    else:
        return pd.DataFrame()

    required = {'predicted_behavior_class'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Inference predictions CSV missing required columns: {missing}')

    out = frame.copy()
    out['predicted_behavior_class'] = pd.to_numeric(out['predicted_behavior_class'], errors='coerce')
    out = out[out['predicted_behavior_class'].isin([-1, 0, 1])].copy()
    out['predicted_behavior_class'] = out['predicted_behavior_class'].astype(int)

    if 'prediction_confidence' in out.columns:
        out['prediction_confidence'] = pd.to_numeric(out['prediction_confidence'], errors='coerce')
    else:
        out['prediction_confidence'] = pd.NA

    if 'loc_state' in out.columns:
        out['loc_state'] = _normalize_text(out['loc_state'])
    else:
        out['loc_state'] = pd.Series(pd.NA, index=out.index, dtype='string')

    if 'device_model' in out.columns:
        out['device_model'] = out['device_model'].astype('string').str.strip()
    else:
        out['device_model'] = pd.Series(pd.NA, index=out.index, dtype='string')

    if 'device_source' in out.columns:
        out['device_source'] = out['device_source'].astype('string').str.strip().fillna('NA')
    else:
        out['device_source'] = pd.Series('NA', index=out.index, dtype='string')

    return out


def get_cluster_payload(state: Optional[str] = None) -> Dict[str, Any]:
    all_clusters = _prepare_cluster_frame()
    clusters = all_clusters
    state_filter = None
    if state is not None and str(state).strip():
        state_filter = str(state).strip().upper()
        if len(state_filter) != 2:
            raise ValueError('State filter must be a 2-letter state code.')
        clusters = clusters[clusters['loc_state'] == state_filter].copy()

    global_max_device_count = int(all_clusters['device_count'].max()) if not all_clusters.empty else 1

    if clusters.empty:
        return {
            'state': state_filter,
            'metric_domain': [-365.0, 365.0],
            'max_device_count': 1,
            'global_max_device_count': global_max_device_count,
            'clusters': [],
        }

    global_metric = all_clusters['remaining_days_weighted_total'].dropna()
    if global_metric.empty:
        metric_min, metric_max = -365.0, 365.0
    else:
        metric_min = float(global_metric.quantile(0.1))
        metric_max = float(global_metric.quantile(0.9))
        if metric_min == metric_max:
            metric_min -= 1.0
            metric_max += 1.0

    max_device_count = int(clusters['device_count'].max())

    serialized_clusters = []
    for row in clusters.itertuples(index=False):
        serialized_clusters.append(
            {
                'cluster_id': row.cluster_id,
                'state': row.loc_state,
                'site_code': row.loc_site_code,
                'site_name': None if pd.isna(row.loc_site_name) else str(row.loc_site_name),
                'latitude': _safe_float(row.loc_latitude),
                'longitude': _safe_float(row.loc_longitude),
                'device_count': _safe_int(row.device_count),
                'eol_known_devices': _safe_int(row.eol_known_devices),
                'overdue_devices': _safe_int(row.overdue_devices),
                'remaining_days_total': _safe_float(row.remaining_days_total),
                'remaining_days_avg': _safe_float(row.remaining_days_avg),
                'remaining_days_weighted_total': _safe_float(row.remaining_days_weighted_total),
            }
        )

    return {
        'state': state_filter,
        'metric_domain': [metric_min, metric_max],
        'max_device_count': max_device_count,
        'global_max_device_count': global_max_device_count,
        'clusters': serialized_clusters,
    }


def get_location_summary(state: str, site_code: str) -> Dict[str, Any]:
    normalized_state = str(state).strip().upper()
    normalized_site = str(site_code).strip().upper()
    if len(normalized_state) != 2:
        raise ValueError('State must be a 2-letter state code.')
    if len(normalized_site) != 3:
        raise ValueError('Site code must be a 3-letter code.')

    frame = _prepare_mappable_device_frame()
    frame['remaining_days'] = compute_days_to_eol(frame['modeldata_eol'])
    location = frame[
        (frame['loc_state'] == normalized_state) & (frame['loc_site_code'] == normalized_site)
    ].copy()

    if location.empty:
        return {
            'state': normalized_state,
            'site_code': normalized_site,
            'found': False,
        }

    source_counts = location['device_source'].astype('string').value_counts().sort_index().to_dict()

    return {
        'state': normalized_state,
        'site_code': normalized_site,
        'site_name': None if location['loc_site_name'].isna().all() else str(location['loc_site_name'].iloc[0]),
        'device_count': int(location['device_name'].nunique()),
        'eol_known_devices': int(location['remaining_days'].count()),
        'overdue_devices': int((location['remaining_days'] < 0).sum()),
        'remaining_days_total': _safe_float(location['remaining_days'].sum(min_count=1)),
        'remaining_days_avg': _safe_float(location['remaining_days'].mean()),
        'sources': source_counts,
        'found': True,
    }


def get_location_summary_with_horizon(state: str, site_code: str, horizon_days: int = 365) -> Dict[str, Any]:
    if horizon_days < 0:
        raise ValueError('horizon_days must be >= 0.')

    base = get_location_summary(state=state, site_code=site_code)
    if not base.get('found'):
        base['horizon_days'] = int(horizon_days)
        base['overdue_count'] = 0
        base['within_horizon_count'] = 0
        base['future_count'] = 0
        base['unknown_count'] = 0
        base['known_count'] = 0
        base['total_devices'] = 0
        return base

    normalized_state = str(state).strip().upper()
    normalized_site = str(site_code).strip().upper()
    frame = _prepare_mappable_device_frame()
    frame['days_to_eol'] = compute_days_to_eol(frame['modeldata_eol'])
    location = frame[
        (frame['loc_state'] == normalized_state) & (frame['loc_site_code'] == normalized_site)
    ].copy()

    bucket_counts = bucket_days_to_eol(location['days_to_eol'], horizon_days=horizon_days)
    if bucket_counts['total_devices'] != int(base['device_count']):
        raise ValueError('Location summary totals are inconsistent with EoL buckets.')

    enriched = dict(base)
    enriched.update(
        {
            'horizon_days': int(horizon_days),
            'total_devices': bucket_counts['total_devices'],
            'overdue_count': bucket_counts['overdue_count'],
            'within_horizon_count': bucket_counts['within_horizon_count'],
            'future_count': bucket_counts['future_count'],
            'unknown_count': bucket_counts['unknown_count'],
            'known_count': bucket_counts['known_count'],
        }
    )
    return enriched


def _prepare_findings_device_frame(horizon_days: int) -> pd.DataFrame:
    frame = _prepare_mappable_device_frame().copy()
    frame['device_source'] = frame['device_source'].astype('string').fillna('NA')
    frame['days_to_eol'] = compute_days_to_eol(frame['modeldata_eol'])

    pricing_cost = pd.to_numeric(frame.get('pricing_total_estimate'), errors='coerce')
    material_cost = pd.to_numeric(frame.get('modeldata_material_cost'), errors='coerce')
    frame['replacement_cost_estimate'] = pricing_cost.fillna(material_cost).fillna(0.0)

    days = frame['days_to_eol']
    frame['eol_bucket'] = np.select(
        [
            days < 0,
            (days >= 0) & (days <= horizon_days),
            days > horizon_days,
            days.isna(),
        ],
        ['overdue', 'within_horizon', 'future', 'unknown'],
        default='unknown',
    )

    frame['support_coverage_score'] = np.where(
        days.isna(),
        30.0,
        np.clip(((days + 365.0) / 1460.0) * 100.0, 0.0, 100.0),
    )

    software_version = frame.get('software_version', pd.Series(index=frame.index, dtype='string'))
    firmware_version = frame.get('firmware_version', pd.Series(index=frame.index, dtype='string'))
    unknown_version_flag = software_version.isna() & firmware_version.isna()

    base_risk = np.select(
        [
            frame['eol_bucket'] == 'overdue',
            frame['eol_bucket'] == 'within_horizon',
            frame['eol_bucket'] == 'future',
            frame['eol_bucket'] == 'unknown',
        ],
        [80.0, 50.0, 20.0, 45.0],
        default=45.0,
    )
    frame['security_risk_score'] = np.clip(
        base_risk
        + (unknown_version_flag.astype(float) * 10.0),
        0.0,
        100.0,
    )

    frame['site_id'] = frame['loc_state'].astype('string') + '-' + frame['loc_site_code'].astype('string')
    frame['near_term_cost_component'] = np.where(
        frame['eol_bucket'].isin(['overdue', 'within_horizon']),
        frame['replacement_cost_estimate'],
        0.0,
    )

    return frame


def get_findings_payload(horizon_days: int = 365) -> Dict[str, Any]:
    if horizon_days < 0:
        raise ValueError('horizon_days must be >= 0.')

    frame = _prepare_findings_device_frame(horizon_days=horizon_days)
    if frame.empty:
        return {
            'horizon_days': int(horizon_days),
            'security_risk_note': (
                'Security risk is a proxy score derived from lifecycle bucket and missing version telemetry.'
            ),
            'kpis': {
                'total_devices': 0,
                'total_sites': 0,
                'overdue_devices': 0,
                'unknown_eol_devices': 0,
                'near_term_cost_estimate': 0.0,
            },
            'top_overdue_sites': [],
            'site_risk_cost_scatter': [],
            'state_exposure': [],
            'source_lifecycle_mix': [],
            'model_hotspots': [],
        }

    site_summary = (
        frame.groupby(['loc_state', 'loc_site_code', 'loc_site_name', 'site_id'], dropna=False, as_index=False)
        .agg(
            total_devices=('device_name', 'size'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            within_horizon_count=('eol_bucket', lambda s: int((s == 'within_horizon').sum())),
            future_count=('eol_bucket', lambda s: int((s == 'future').sum())),
            unknown_count=('eol_bucket', lambda s: int((s == 'unknown').sum())),
            eol_known_devices=('days_to_eol', 'count'),
            avg_days_to_eol=('days_to_eol', 'mean'),
            support_coverage_score=('support_coverage_score', 'mean'),
            security_risk_score=('security_risk_score', 'mean'),
            replacement_cost_total=('replacement_cost_estimate', 'sum'),
            near_term_cost=('near_term_cost_component', 'sum'),
        )
        .sort_values(['overdue_count', 'near_term_cost', 'total_devices'], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    site_summary['overdue_rate'] = np.where(
        site_summary['total_devices'] > 0,
        (site_summary['overdue_count'] / site_summary['total_devices']) * 100.0,
        0.0,
    )

    top_overdue = site_summary.head(10)
    top_overdue_sites = [
        {
            'state': _safe_str(row.loc_state),
            'site_code': _safe_str(row.loc_site_code),
            'site_name': _safe_str(row.loc_site_name),
            'site_id': _safe_str(row.site_id),
            'total_devices': _safe_int(row.total_devices),
            'overdue_count': _safe_int(row.overdue_count),
            'overdue_rate': _safe_float(row.overdue_rate),
            'near_term_cost': _safe_float(row.near_term_cost),
        }
        for row in top_overdue.itertuples(index=False)
    ]

    scatter_sites = site_summary.sort_values(
        ['total_devices', 'overdue_count', 'near_term_cost'],
        ascending=[False, False, False],
    ).head(350)
    site_risk_cost_scatter = [
        {
            'site_id': _safe_str(row.site_id),
            'site_name': _safe_str(row.loc_site_name),
            'state': _safe_str(row.loc_state),
            'site_code': _safe_str(row.loc_site_code),
            'total_devices': _safe_int(row.total_devices),
            'support_coverage_score': _safe_float(row.support_coverage_score),
            'security_risk_score': _safe_float(row.security_risk_score),
            'replacement_cost_total': _safe_float(row.replacement_cost_total),
            'near_term_cost': _safe_float(row.near_term_cost),
            'overdue_rate': _safe_float(row.overdue_rate),
        }
        for row in scatter_sites.itertuples(index=False)
    ]

    state_summary = (
        frame.groupby('loc_state', as_index=False)
        .agg(
            total_devices=('device_name', 'size'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            unknown_count=('eol_bucket', lambda s: int((s == 'unknown').sum())),
            support_coverage_score=('support_coverage_score', 'mean'),
            security_risk_score=('security_risk_score', 'mean'),
            near_term_cost=('near_term_cost_component', 'sum'),
        )
        .sort_values(['near_term_cost', 'overdue_count'], ascending=[False, False])
        .head(15)
        .reset_index(drop=True)
    )
    state_exposure = [
        {
            'state': _safe_str(row.loc_state),
            'total_devices': _safe_int(row.total_devices),
            'overdue_count': _safe_int(row.overdue_count),
            'unknown_count': _safe_int(row.unknown_count),
            'support_coverage_score': _safe_float(row.support_coverage_score),
            'security_risk_score': _safe_float(row.security_risk_score),
            'near_term_cost': _safe_float(row.near_term_cost),
        }
        for row in state_summary.itertuples(index=False)
    ]

    source_summary = (
        frame.groupby('device_source', as_index=False)
        .agg(
            total_devices=('device_name', 'size'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            within_horizon_count=('eol_bucket', lambda s: int((s == 'within_horizon').sum())),
            future_count=('eol_bucket', lambda s: int((s == 'future').sum())),
            unknown_count=('eol_bucket', lambda s: int((s == 'unknown').sum())),
        )
        .sort_values('total_devices', ascending=False)
        .reset_index(drop=True)
    )
    source_lifecycle_mix = [
        {
            'device_source': _safe_str(row.device_source),
            'total_devices': _safe_int(row.total_devices),
            'overdue_count': _safe_int(row.overdue_count),
            'within_horizon_count': _safe_int(row.within_horizon_count),
            'future_count': _safe_int(row.future_count),
            'unknown_count': _safe_int(row.unknown_count),
        }
        for row in source_summary.itertuples(index=False)
    ]

    model_summary = (
        frame[frame['device_model'].notna()]
        .groupby('device_model', as_index=False)
        .agg(
            total_devices=('device_name', 'size'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            near_term_cost=('near_term_cost_component', 'sum'),
            security_risk_score=('security_risk_score', 'mean'),
        )
        .sort_values(['overdue_count', 'near_term_cost', 'total_devices'], ascending=[False, False, False])
        .head(12)
        .reset_index(drop=True)
    )
    model_hotspots = [
        {
            'device_model': _safe_str(row.device_model),
            'total_devices': _safe_int(row.total_devices),
            'overdue_count': _safe_int(row.overdue_count),
            'near_term_cost': _safe_float(row.near_term_cost),
            'security_risk_score': _safe_float(row.security_risk_score),
        }
        for row in model_summary.itertuples(index=False)
    ]

    kpis = {
        'total_devices': int(site_summary['total_devices'].sum()),
        'total_sites': int(site_summary.shape[0]),
        'overdue_devices': int(site_summary['overdue_count'].sum()),
        'unknown_eol_devices': int(site_summary['unknown_count'].sum()),
        'near_term_cost_estimate': float(site_summary['near_term_cost'].sum()),
    }

    return {
        'horizon_days': int(horizon_days),
        'security_risk_note': (
            'Security risk is a proxy score derived from lifecycle bucket and missing version telemetry.'
        ),
        'kpis': kpis,
        'top_overdue_sites': top_overdue_sites,
        'site_risk_cost_scatter': site_risk_cost_scatter,
        'state_exposure': state_exposure,
        'source_lifecycle_mix': source_lifecycle_mix,
        'model_hotspots': model_hotspots,
    }


def get_inferences_payload() -> Dict[str, Any]:
    frame = load_inference_predictions()
    if frame.empty:
        return {
            'available': False,
            'note': (
                'No inference prediction file found. Expected '
                '`data/unknown_eol_device_behavior_predictions.csv` or '
                '`data/device_behavior_predictions.csv`.'
            ),
            'artifacts': {
                'model_performance_image': '/reports/model_performance.png',
                'model_performance_exists': MODEL_PERFORMANCE_PNG.exists(),
                'model_pkl_path': str(RANDOM_FOREST_MODEL_PKL),
                'model_pkl_exists': RANDOM_FOREST_MODEL_PKL.exists(),
            },
            'kpis': {},
            'class_mix': [],
            'state_risk': [],
            'model_risk': [],
            'source_mix': [],
            'confidence_distribution': [],
        }

    total_scored = int(frame.shape[0])
    passed_count = int((frame['predicted_behavior_class'] == -1).sum())
    within_count = int((frame['predicted_behavior_class'] == 1).sum())
    default_count = int((frame['predicted_behavior_class'] == 0).sum())
    risk_total = passed_count + within_count
    risk_rate = (risk_total / total_scored) * 100.0 if total_scored > 0 else 0.0

    class_label_map = {
        -1: 'Likely Passed EoL',
        1: 'Likely EoL <= 365d',
        0: 'Likely EoL > 365d',
    }
    class_order = [-1, 1, 0]
    class_mix = []
    for cls in class_order:
        count = int((frame['predicted_behavior_class'] == cls).sum())
        class_mix.append(
            {
                'class_id': int(cls),
                'label': class_label_map[cls],
                'count': count,
                'share_pct': (count / total_scored) * 100.0 if total_scored > 0 else 0.0,
            }
        )

    frame['is_predicted_risk'] = frame['predicted_behavior_class'].isin([-1, 1])

    state_risk_df = (
        frame[frame['loc_state'].notna()]
        .groupby('loc_state', as_index=False)
        .agg(
            total_devices=('predicted_behavior_class', 'size'),
            predicted_risk_devices=('is_predicted_risk', 'sum'),
            predicted_passed_eol=('predicted_behavior_class', lambda s: int((s == -1).sum())),
            predicted_within_365=('predicted_behavior_class', lambda s: int((s == 1).sum())),
        )
        .sort_values(['predicted_risk_devices', 'predicted_passed_eol', 'total_devices'], ascending=[False, False, False])
        .head(15)
        .reset_index(drop=True)
    )
    state_risk_df['predicted_risk_rate'] = np.where(
        state_risk_df['total_devices'] > 0,
        (state_risk_df['predicted_risk_devices'] / state_risk_df['total_devices']) * 100.0,
        0.0,
    )
    state_risk = [
        {
            'state': _safe_str(row.loc_state),
            'total_devices': _safe_int(row.total_devices),
            'predicted_risk_devices': _safe_int(row.predicted_risk_devices),
            'predicted_passed_eol': _safe_int(row.predicted_passed_eol),
            'predicted_within_365': _safe_int(row.predicted_within_365),
            'predicted_risk_rate': _safe_float(row.predicted_risk_rate),
        }
        for row in state_risk_df.itertuples(index=False)
    ]

    model_risk_df = (
        frame[frame['device_model'].notna()]
        .groupby('device_model', as_index=False)
        .agg(
            total_devices=('predicted_behavior_class', 'size'),
            predicted_risk_devices=('is_predicted_risk', 'sum'),
            predicted_passed_eol=('predicted_behavior_class', lambda s: int((s == -1).sum())),
            predicted_within_365=('predicted_behavior_class', lambda s: int((s == 1).sum())),
        )
        .sort_values(['predicted_risk_devices', 'predicted_passed_eol', 'total_devices'], ascending=[False, False, False])
        .head(12)
        .reset_index(drop=True)
    )
    model_risk = [
        {
            'device_model': _safe_str(row.device_model),
            'total_devices': _safe_int(row.total_devices),
            'predicted_risk_devices': _safe_int(row.predicted_risk_devices),
            'predicted_passed_eol': _safe_int(row.predicted_passed_eol),
            'predicted_within_365': _safe_int(row.predicted_within_365),
        }
        for row in model_risk_df.itertuples(index=False)
    ]

    source_mix_df = (
        frame.groupby('device_source', as_index=False)
        .agg(
            total_devices=('predicted_behavior_class', 'size'),
            predicted_passed_eol=('predicted_behavior_class', lambda s: int((s == -1).sum())),
            predicted_within_365=('predicted_behavior_class', lambda s: int((s == 1).sum())),
            predicted_gt_365=('predicted_behavior_class', lambda s: int((s == 0).sum())),
        )
        .sort_values('total_devices', ascending=False)
        .reset_index(drop=True)
    )
    source_mix = [
        {
            'device_source': _safe_str(row.device_source),
            'total_devices': _safe_int(row.total_devices),
            'predicted_passed_eol': _safe_int(row.predicted_passed_eol),
            'predicted_within_365': _safe_int(row.predicted_within_365),
            'predicted_gt_365': _safe_int(row.predicted_gt_365),
        }
        for row in source_mix_df.itertuples(index=False)
    ]

    confidence_distribution = []
    confidence = pd.to_numeric(frame['prediction_confidence'], errors='coerce')
    confidence = confidence[confidence.notna()]
    if not confidence.empty:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        bucketed = pd.cut(confidence.clip(lower=0.0, upper=1.0), bins=bins, labels=labels, include_lowest=True)
        bucket_counts = bucketed.value_counts().reindex(labels, fill_value=0)
        confidence_distribution = [
            {'bin': str(bin_label), 'count': int(count)}
            for bin_label, count in bucket_counts.items()
        ]

    return {
        'available': True,
        'note': (
            'Inference results are generated from unknown-EoL devices and predicted into '
            'three behavior classes: passed EoL, within 12 months, and beyond 12 months.'
        ),
        'artifacts': {
            'model_performance_image': '/reports/model_performance.png',
            'model_performance_exists': MODEL_PERFORMANCE_PNG.exists(),
            'model_pkl_path': str(RANDOM_FOREST_MODEL_PKL),
            'model_pkl_exists': RANDOM_FOREST_MODEL_PKL.exists(),
        },
        'kpis': {
            'total_scored': total_scored,
            'predicted_passed_eol': passed_count,
            'predicted_within_365': within_count,
            'predicted_gt_365': default_count,
            'predicted_risk_total': risk_total,
            'predicted_risk_rate': risk_rate,
        },
        'class_mix': class_mix,
        'state_risk': state_risk,
        'model_risk': model_risk,
        'source_mix': source_mix,
        'confidence_distribution': confidence_distribution,
    }
