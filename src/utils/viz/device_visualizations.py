from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.utils.data import compute_days_to_eol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / 'reports'
DEFAULT_HORIZON_DAYS = 365
EOL_BUCKET_ORDER = ['overdue', 'within_horizon', 'future', 'unknown']
EOL_BUCKET_LABELS = {
    'overdue': 'Overdue',
    'within_horizon': f'Within {DEFAULT_HORIZON_DAYS}d',
    'future': 'Future',
    'unknown': 'Unknown',
}
EOL_COLORS = {
    'overdue': '#d1495b',
    'within_horizon': '#f2c14e',
    'future': '#2a9d50',
    'unknown': '#9ea8ae',
}


def _safe_series(frame: pd.DataFrame, column_name: str, default_value: Any = pd.NA) -> pd.Series:
    if column_name in frame.columns:
        return frame[column_name]
    
    return pd.Series(default_value, index=frame.index)


def _save_figure(
    fig: plt.Figure,
    savefig: bool,
    filename: Optional[str],
    reports_dir: str | Path,
    dpi: int = 300,
) -> Optional[Path]:
    if not savefig:
        return None

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = filename or 'figure.png'
    output_path = output_dir / output_name
    if output_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}:
        output_path = output_path.with_suffix('.png')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    return output_path


def _render_streamlit(fig: plt.Figure, use_streamlit: bool, streamlit_container: Any = None) -> None:
    if not use_streamlit:
        return
    if st is None:
        raise RuntimeError('Streamlit is not available in the current environment.')
    
    target = streamlit_container if streamlit_container is not None else st
    target.pyplot(fig, width='stretch')


def _finalize_figure(
    fig: plt.Figure,
    savefig: bool,
    filename: Optional[str],
    reports_dir: str | Path,
    use_streamlit: bool,
    streamlit_container: Any = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, Optional[Path]]:
    fig.tight_layout()
    saved_path = _save_figure(fig, savefig=savefig, filename=filename, reports_dir=reports_dir, dpi=dpi)
    _render_streamlit(fig, use_streamlit=use_streamlit, streamlit_container=streamlit_container)

    return fig, saved_path


def _prepare_analytics_frame(device_dataset: pd.DataFrame, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    if 'device_name' not in device_dataset.columns:
        raise ValueError("Expected column 'device_name' in device dataset.")

    frame = device_dataset.copy()
    frame['device_source'] = _safe_series(frame, 'device_source').astype('string').str.strip()
    frame['device_source'] = frame['device_source'].replace({'': pd.NA, 'NAN': pd.NA}).fillna('NA')

    modeldata_eol = _safe_series(frame, 'modeldata_eol', pd.NaT)
    frame['days_to_eol'] = compute_days_to_eol(modeldata_eol)

    days = frame['days_to_eol']
    frame['eol_bucket'] = np.select(
        [
            days < 0,
            (days >= 0) & (days <= horizon_days),
            days > horizon_days,
            days.isna()
        ],
        ['overdue', 'within_horizon', 'future', 'unknown'],
        default='unknown'
    )

    pricing_cost = pd.to_numeric(_safe_series(frame, 'pricing_total_estimate'), errors='coerce')
    material_cost = pd.to_numeric(_safe_series(frame, 'modeldata_material_cost'), errors='coerce')
    labor_cost = pd.to_numeric(_safe_series(frame, 'modeldata_labor_cost'), errors='coerce')
    frame['replacement_cost_estimate'] = pricing_cost.fillna(material_cost).fillna(0.0)
    frame['labor_cost_estimate'] = labor_cost.fillna(0.0)
    frame['near_term_cost_component'] = np.where(
        frame['eol_bucket'].isin(['overdue', 'within_horizon']),
        frame['replacement_cost_estimate'],
        0.0
    )

    frame['support_coverage_score'] = np.where(
        days.isna(),
        30.0,
        np.clip(((days + 365.0) / 1460.0) * 100.0, 0.0, 100.0)
    )

    software_version = _safe_series(frame, 'software_version')
    firmware_version = _safe_series(frame, 'firmware_version')
    unknown_version_flag = software_version.isna() & firmware_version.isna()

    base_risk = np.select(
        [
            frame['eol_bucket'] == 'overdue',
            frame['eol_bucket'] == 'within_horizon',
            frame['eol_bucket'] == 'future',
            frame['eol_bucket'] == 'unknown',
        ],
        [80.0, 50.0, 20.0, 45.0],
        default=45.0
    )
    frame['security_risk_score'] = np.clip(
        base_risk + (unknown_version_flag.astype(float) * 10.0),
        0.0,
        100.0,
    )

    frame['loc_state'] = _safe_series(frame, 'loc_state').astype('string').str.strip().str.upper()
    frame['loc_site_code'] = _safe_series(frame, 'loc_site_code').astype('string').str.strip().str.upper()
    frame['loc_site_name'] = _safe_series(frame, 'loc_site_name').astype('string')
    frame['loc_owner'] = _safe_series(frame, 'loc_owner').astype('string')
    frame['loc_call_group'] = _safe_series(frame, 'loc_call_group').astype('string')
    frame['modeldata_category'] = _safe_series(frame, 'modeldata_category').astype('string')
    frame['device_model'] = _safe_series(frame, 'device_model').astype('string')
    frame['site_id'] = frame['loc_state'].fillna('NA') + '-' + frame['loc_site_code'].fillna('NA')

    return frame


def plot_eol_bucket_distribution(
    device_dataset: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_eol_bucket_distribution.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    counts = frame['eol_bucket'].value_counts().reindex(EOL_BUCKET_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [EOL_BUCKET_LABELS[b] if b != 'within_horizon' else f"Within {horizon_days}d" for b in counts.index]
    ax.bar(labels, counts.values, color=[EOL_COLORS[b] for b in counts.index])
    ax.set_title("Lifecycle Status Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Device Count")
    ax.grid(axis="y", alpha=0.25)
    for idx, val in enumerate(counts.values):
        ax.text(idx, val, f'{int(val):,}', ha='center', va='bottom', fontsize=9)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_days_to_eol_distribution(
    device_dataset: pd.DataFrame,
    bins: int = 40,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_days_to_eol_distribution.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    known = frame['days_to_eol'].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(known, bins=bins, kde=True, color='#3f7cac', ax=ax)
    ax.axvline(0, color=EOL_COLORS['overdue'], linestyle='--', linewidth=2, label="EoL threshold (0)")
    ax.axvline(horizon_days, color=EOL_COLORS['within_horizon'], linestyle='--', linewidth=2, label=f"Horizon ({horizon_days}d)")
    ax.set_title("Distribution of Days to EoL (Known Values)")
    ax.set_xlabel("Days to EoL")
    ax.set_ylabel("Devices")
    ax.legend(loc='upper right')
    ax.grid(axis="y", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_overdue_age_distribution(
    device_dataset: pd.DataFrame,
    bins: int = 35,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_overdue_age_distribution.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    overdue_age = (-frame.loc[frame['days_to_eol'] < 0, 'days_to_eol']).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(overdue_age, bins=bins, color=EOL_COLORS['overdue'], ax=ax)
    ax.set_title("How Long Devices Have Been Overdue")
    ax.set_xlabel("Days Overdue")
    ax.set_ylabel("Devices")
    ax.grid(axis="y", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_top_overdue_sites(
    device_dataset: pd.DataFrame,
    top_n: int = 10,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_top_overdue_sites.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    sites = (
        frame.dropna(
            subset=['loc_state', 'loc_site_code']
        ).groupby(
            ['loc_state', 'loc_site_code', 'loc_site_name'],
            as_index=False
        ).agg(
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            total_devices=('device_name', 'size'),
            near_term_cost=('near_term_cost_component', 'sum'),
        ).sort_values(
            ['overdue_count', 'near_term_cost', 'total_devices'],
            ascending=[False, False, False]
        ).head(top_n)
    )
    sites['site_label'] = sites['loc_state'].astype(str) + '-' + sites['loc_site_code'].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=sites,
        y='site_label',
        x='overdue_count',
        color=EOL_COLORS['overdue'],
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Sites by Overdue Device Count")
    ax.set_xlabel("Overdue Devices")
    ax.set_ylabel("Site")
    for idx, (_, row) in enumerate(sites.iterrows()):
        ax.text(row['overdue_count'], idx, f"  {int(row['total_devices'])} total", va='center', fontsize=8)

    ax.grid(axis="x", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_state_exposure(
    device_dataset: pd.DataFrame,
    top_n: int = 15,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_state_exposure_near_term_cost.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    states = (
        frame.dropna(subset=['loc_state']).groupby(
            'loc_state', as_index=False
        ).agg(
            near_term_cost=('near_term_cost_component', 'sum'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
        ).sort_values(
            'near_term_cost', ascending=False
        ).head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=states, y='loc_state', x='near_term_cost', color='#365f9c', ax=ax)
    ax.set_title(f"Top {top_n} States by Near-Term Replacement Cost Exposure")
    ax.set_xlabel("Near-Term Cost Estimate")
    ax.set_ylabel("State")
    for idx, (_, row) in enumerate(states.iterrows()):
        ax.text(row['near_term_cost'], idx, f"  overdue={int(row['overdue_count'])}", va='center', fontsize=8)

    ax.grid(axis="x", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_source_lifecycle_mix(
    device_dataset: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    normalize: bool = True,
    savefig: bool = False,
    filename: str = 'viz_source_lifecycle_mix.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    pivot = pd.crosstab(frame['device_source'], frame['eol_bucket']).reindex(columns=EOL_BUCKET_ORDER, fill_value=0)
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(pivot.shape[0], dtype=float)
    x = np.arange(pivot.shape[0])
    for bucket in EOL_BUCKET_ORDER:
        values = pivot[bucket].values
        ax.bar(x, values, bottom=bottom, color=EOL_COLORS[bucket], label=bucket.replace('_', ' ').title())
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=20)
    ax.set_ylabel("Share of Devices" if normalize else "Device Count")
    ax.set_title("Lifecycle Mix by Source System")
    ax.legend(loc='upper right')
    ax.grid(axis="y", alpha=0.25)
    
    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_category_lifecycle_heatmap(
    device_dataset: pd.DataFrame,
    top_n: int = 12,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_category_lifecycle_heatmap.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    categories = frame['modeldata_category'].fillna('Unknown')
    heat = pd.crosstab(categories, frame['eol_bucket']).reindex(columns=EOL_BUCKET_ORDER, fill_value=0)
    heat = heat.sort_values('overdue', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.3, cbar=True, ax=ax)
    ax.set_title("Lifecycle Pressure by Device Category")
    ax.set_xlabel("Lifecycle Bucket")
    ax.set_ylabel("Category")
    
    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_model_hotspots(
    device_dataset: pd.DataFrame,
    top_n: int = 12,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_model_hotspots.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    models = (
        frame.dropna(subset=['device_model']).groupby(
            'device_model', as_index=False
        ).agg(
            total_devices=('device_name', 'size'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            near_term_cost=('near_term_cost_component', 'sum'),
        )
    )
    models = models.sort_values(['overdue_count', 'near_term_cost', 'total_devices'], ascending=[False, False, False]).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=models, y='device_model', x='overdue_count', color='#8d3fbc', ax=ax)
    ax.set_title(f"Top {top_n} Model Hotspots by Overdue Count")
    ax.set_xlabel("Overdue Devices")
    ax.set_ylabel("Device Model")
    for idx, (_, row) in enumerate(models.iterrows()):
        ax.text(row['overdue_count'], idx, f"  ${row['near_term_cost']:,.0f}", va='center', fontsize=8)\
        
    ax.grid(axis="x", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_lifecycle_risk_cost_scatter(
    device_dataset: pd.DataFrame,
    max_sites: int = 500,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_lifecycle_risk_cost_scatter.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    sites = (
        frame.dropna(
            subset=['loc_state', 'loc_site_code']
        ).groupby(
            ['site_id', 'loc_site_name', 'loc_state', 'loc_site_code'],
            as_index=False
        ).agg(
            total_devices=('device_name', 'size'),
            support_coverage_score=('support_coverage_score', 'mean'),
            security_risk_score=('security_risk_score', 'mean'),
            near_term_cost=('near_term_cost_component', 'sum'),
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
        ).sort_values(
            ['total_devices', 'near_term_cost'],
            ascending=[False, False]
        ).head(max_sites)
    )
    sites['overdue_rate'] = np.where(sites['total_devices'] > 0, (sites['overdue_count'] / sites['total_devices']) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        sites['support_coverage_score'],
        sites['security_risk_score'],
        s=np.clip(20 + np.sqrt(sites['near_term_cost'].fillna(0)) / 20, 18, 220),
        c=sites['overdue_rate'],
        cmap='RdYlGn_r',
        alpha=0.75,
        edgecolor='#1f2b25',
        linewidth=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Overdue Rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Support Coverage Score (Higher = Better)")
    ax.set_ylabel("Security Risk Proxy (Higher = Riskier)")
    ax.set_title("Lifecycle Status vs Support Coverage, Security Risk, and Cost")
    ax.grid(alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_owner_risk_boxplot(
    device_dataset: pd.DataFrame,
    top_n: int = 8,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_owner_risk_boxplot.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    owners = frame['loc_owner'].fillna('Unknown')
    top_owners = owners.value_counts().head(top_n).index
    plot_frame = frame[owners.isin(top_owners)].copy()
    plot_frame['loc_owner'] = plot_frame['loc_owner'].fillna('Unknown')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_frame, x='loc_owner', y='security_risk_score', color='#72a98f', ax=ax)
    ax.set_title("Security Risk Proxy Distribution by Site Owner")
    ax.set_xlabel("Owner")
    ax.set_ylabel("Security Risk Proxy")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_call_group_backlog(
    device_dataset: pd.DataFrame,
    top_n: int = 12,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_call_group_backlog.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    call_groups = (
        frame.groupby(
            'loc_call_group', as_index=False
        ).agg(
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            within_count=('eol_bucket', lambda s: int((s == 'within_horizon').sum())),
            total_devices=('device_name', 'size'),
        ).sort_values(
            ['overdue_count', 'within_count', 'total_devices'],
            ascending=[False, False, False]
        ).head(top_n)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(call_groups['loc_call_group'], call_groups['within_count'], color=EOL_COLORS['within_horizon'], label=f"Within {horizon_days}d")
    ax.barh(call_groups['loc_call_group'], call_groups['overdue_count'], left=call_groups['within_count'], color=EOL_COLORS['overdue'], label="Overdue")
    ax.set_title("Field Operations Call Group Backlog")
    ax.set_xlabel("Devices Requiring Near-Term Action")
    ax.set_ylabel("Call Group")
    ax.legend(loc='lower right')
    ax.grid(axis="x", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_unknown_eol_by_state(
    device_dataset: pd.DataFrame,
    top_n: int = 15,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_unknown_eol_by_state.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    states = (
        frame.dropna(subset=['loc_state']).groupby(
            'loc_state', as_index=False
        ).agg(
            total_devices=('device_name', 'size'),
            unknown_count=('eol_bucket', lambda s: int((s == 'unknown').sum())),
        ).sort_values(
            'unknown_count', ascending=False
        ).head(top_n)
    )
    states['unknown_rate'] = np.where(states['total_devices'] > 0, (states['unknown_count'] / states['total_devices']) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=states, y="loc_state", x="unknown_rate", color=EOL_COLORS['unknown'], ax=ax)
    ax.set_title("Top States by Unknown EoL Rate")
    ax.set_xlabel("Unknown EoL Rate (%)")
    ax.set_ylabel("State")
    ax.grid(axis="x", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_cost_composition_by_lifecycle_bucket(
    device_dataset: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_cost_composition_by_bucket.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    agg = (
        frame.groupby('eol_bucket', as_index=False).agg(
            replacement_cost=('replacement_cost_estimate', 'sum'),
            labor_cost=('labor_cost_estimate', 'sum'),
        ).set_index('eol_bucket').reindex(EOL_BUCKET_ORDER, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(agg.index))
    ax.bar(x, agg['replacement_cost'].values, color='#4378bf', label="Replacement Cost")
    ax.bar(x, agg['labor_cost'].values, bottom=agg['replacement_cost'].values, color='#7fb069', label="Labor Cost")
    ax.set_xticks(x)
    ax.set_xticklabels([EOL_BUCKET_LABELS.get(k, k.title()) for k in agg.index], rotation=15)
    ax.set_title("Cost Composition by Lifecycle Bucket")
    ax.set_ylabel("Estimated Cost")
    ax.legend(loc='upper right')
    ax.grid(axis="y", alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def plot_site_priority_matrix(
    device_dataset: pd.DataFrame,
    top_n: int = 250,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    filename: str = 'viz_site_priority_matrix.png',
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> Tuple[plt.Figure, Optional[Path]]:
    frame = _prepare_analytics_frame(device_dataset, horizon_days=horizon_days)
    sites = (
        frame.dropna(subset=['loc_state', 'loc_site_code']).groupby(
            ['site_id', 'loc_site_name'], as_index=False
        ).agg(
            overdue_count=('eol_bucket', lambda s: int((s == 'overdue').sum())),
            within_count=('eol_bucket', lambda s: int((s == 'within_horizon').sum())),
            unknown_count=('eol_bucket', lambda s: int((s == 'unknown').sum())),
            near_term_cost=('near_term_cost_component', 'sum'),
            security_risk_score=('security_risk_score', 'mean'),
        )
    )
    sites['action_backlog'] = sites['overdue_count'] + sites['within_count']
    sites = sites.sort_values(['action_backlog', 'near_term_cost'], ascending=[False, False]).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sites['action_backlog'],
        sites['near_term_cost'],
        s=np.clip(30 + np.sqrt(sites['unknown_count'].fillna(0) + 1) * 12, 24, 220),
        c=sites['security_risk_score'],
        cmap='magma_r',
        alpha=0.75,
        edgecolor='#1f2b25',
        linewidth=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Security Risk Proxy")
    ax.set_title("Site Priority Matrix (Backlog vs Near-Term Cost)")
    ax.set_xlabel("Action Backlog (Overdue + Within Horizon)")
    ax.set_ylabel("Near-Term Cost Estimate")
    ax.grid(alpha=0.25)

    return _finalize_figure(fig, savefig, filename, reports_dir, use_streamlit, streamlit_container)


def generate_appendix_visual_suite(
    device_dataset: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    savefig: bool = False,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    use_streamlit: bool = False,
    streamlit_container: Any = None,
) -> dict[str, tuple[plt.Figure, Optional[Path]]]:
    """
    Generate a broad visualization suite for technical appendix and Q&A prep.
    """
    return {
        'eol_bucket_distribution': plot_eol_bucket_distribution(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_eol_bucket_distribution.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'days_to_eol_distribution': plot_days_to_eol_distribution(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_days_to_eol_distribution.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'overdue_age_distribution': plot_overdue_age_distribution(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_overdue_age_distribution.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'top_overdue_sites': plot_top_overdue_sites(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_top_overdue_sites.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'state_exposure': plot_state_exposure(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_state_exposure.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'source_lifecycle_mix': plot_source_lifecycle_mix(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_source_lifecycle_mix.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'category_lifecycle_heatmap': plot_category_lifecycle_heatmap(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_category_lifecycle_heatmap.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'model_hotspots': plot_model_hotspots(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_model_hotspots.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'risk_cost_scatter': plot_lifecycle_risk_cost_scatter(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_risk_cost_scatter.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'owner_risk_boxplot': plot_owner_risk_boxplot(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_owner_risk_boxplot.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'call_group_backlog': plot_call_group_backlog(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_call_group_backlog.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'unknown_eol_by_state': plot_unknown_eol_by_state(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_unknown_eol_by_state.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'cost_composition': plot_cost_composition_by_lifecycle_bucket(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_cost_composition.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
        'site_priority_matrix': plot_site_priority_matrix(
            device_dataset,
            horizon_days=horizon_days,
            savefig=savefig,
            filename='appendix_site_priority_matrix.png',
            reports_dir=reports_dir,
            use_streamlit=use_streamlit,
            streamlit_container=streamlit_container,
        ),
    }
