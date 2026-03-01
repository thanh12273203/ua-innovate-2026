from __future__ import annotations

from pathlib import Path
from typing import Dict
import re

import pandas as pd


def wrangle(excel_filepath: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load all worksheets from an Excel workbook into a dictionary of DataFrames.

    Parameters
    ----------
    excel_filepath : str or pathlib.Path
        Path to the Excel (.xlsx, .xlsm, .xls) file.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary where keys are worksheet (tab) names and values are
        the corresponding DataFrames.
    """
    excel_path = Path(excel_filepath)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if excel_path.suffix.lower() not in {'.xlsx', '.xlsm', '.xls'}:
        raise ValueError(f"Unsupported Excel extension: {excel_path.suffix}")

    excel_file = pd.ExcelFile(excel_path)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in excel_file.sheet_names:
        data_frame = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            dtype=object,
            engine=None,
        )
        sheets[sheet_name] = data_frame

    return sheets


def apply_filter(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Apply a filter to each DataFrame in the dictionary of sheets.

    Parameters
    ----------
    sheets : Dict[str, pd.DataFrame]
        Dictionary where keys are worksheet (tab) names and values are
        the corresponding DataFrames.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with filtered DataFrames.
    """
    if ['SOLID', 'SOLID-Loc', 'NA', 'PrimeAP', 'PrimeWLC', 'CatCtr', 'Decom', 'ModelData', 'Pricing', 'Glossary'] != list(sheets.keys()):
        raise ValueError("The provided sheets dictionary does not contain the expected sheet names.")
    
    # Filter out decommissioned sites
    print(f"Decommissioned sites in 'SOLID' sheet: {sheets['SOLID']['Site Code'].isin(sheets['Decom']['Site Cd']).sum()}")
    sheets['SOLID'] = sheets['SOLID'][~sheets['SOLID']['Site Code'].isin(sheets['Decom']['Site Cd'])]
    sheets['SOLID-Loc'] = sheets['SOLID-Loc'][~sheets['SOLID-Loc']['Site Code'].isin(sheets['Decom']['Site Cd'])]
    
    # Edit the 'hostname' column in the 'CatCtr' sheet to remove the domain part
    sheets['CatCtr']['hostname'] = sheets['CatCtr']['hostname'].str.split('.').str[0]

    # Filter out the unreachable devices in the 'CatCtr' sheet
    print(f"\nUnreachable devices in 'CatCtr' sheet: {sheets['CatCtr']['reachabilityStatus'].isin(['Unreachable', 'Ping Reachable']).sum()}")
    sheets['CatCtr'] = sheets['CatCtr'][~sheets['CatCtr']['reachabilityStatus'].isin(['Unreachable', 'Ping Reachable'])]

    # Filter out the unsupported devices with no hostname in the 'CatCtr' sheet
    print(f"\nDevices in 'CatCtr' sheet with no hostname: {sheets['CatCtr']['hostname'].isna().sum()}")
    sheets['CatCtr'] = sheets['CatCtr'][sheets['CatCtr']['hostname'].notna()]

    # Edit the device family in the 'CatCtr' sheet
    sheets['CatCtr']['family'] = sheets['CatCtr']['family'].replace(
        to_replace={
            'Unified AP': 'AP',
            'Routers': 'Router',
            'Switches and Hubs': 'Switch'
        }
    )

    # Filter out device names in the 'NA', 'PrimeAP', and 'PrimeWLC' sheets that are present in the 'CatCtr' sheet
    print(f"\nDevices in 'NA' sheet that are present in 'CatCtr' sheet: {sheets['NA']['Host Name'].isin(sheets['CatCtr']['hostname']).sum()}")
    print(f"Devices in 'PrimeAP' sheet that are present in 'CatCtr' sheet: {sheets['PrimeAP']['name'].isin(sheets['CatCtr']['hostname']).sum()}")
    print(f"Devices in 'PrimeWLC' sheet that are present in 'CatCtr' sheet: {sheets['PrimeWLC']['deviceName'].isin(sheets['CatCtr']['hostname']).sum()}")
    sheets['NA'] = sheets['NA'][~sheets['NA']['Host Name'].isin(sheets['CatCtr']['hostname'])]
    sheets['PrimeAP'] = sheets['PrimeAP'][~sheets['PrimeAP']['name'].isin(sheets['CatCtr']['hostname'])]
    sheets['PrimeWLC'] = sheets['PrimeWLC'][~sheets['PrimeWLC']['deviceName'].isin(sheets['CatCtr']['hostname'])]

    # Filter out device names in the 'NA' sheet that are present in 'PrimeAP' and 'PrimeWLC' sheets
    print(f"\nDevices in 'NA' sheet that are present in 'PrimeAP' sheet: {sheets['NA']['Host Name'].isin(sheets['PrimeAP']['name']).sum()}")
    print(f"Devices in 'NA' sheet that are present in 'PrimeWLC' sheet: {sheets['NA']['Host Name'].isin(sheets['PrimeWLC']['deviceName']).sum()}")
    sheets['NA'] = sheets['NA'][~sheets['NA']['Host Name'].isin(sheets['PrimeAP']['name'])]
    sheets['NA'] = sheets['NA'][~sheets['NA']['Host Name'].isin(sheets['PrimeWLC']['deviceName'])]

    # Filter out inactive devices in the 'NA' sheet
    print(f"\nInactive devices in 'NA' sheet: {sheets['NA']['Device Status'].isin(['Inactive']).sum()}")
    sheets['NA'] = sheets['NA'][sheets['NA']['Device Status'] != 'Inactive']

    # Select only the relevant 'Device Type' values in the 'NA' sheet
    print(
        f"Number of devices filtered out in 'NA' sheet based on 'Device Type': {sheets['NA']['Device Type'].isin(
        ['Wireless Controller', 'Firewall', 'Virtual Firewall', 'WirelessLC']
        ).sum()}"
    )
    sheets['NA'] = sheets['NA'][~sheets['NA']['Device Type'].isin(
        ['Wireless Controller', 'Firewall', 'Virtual Firewall', 'WirelessLC']
    )]

    # Edit the 'Device Type' containing 'Switch' to 'Switch' in the 'NA' sheet
    sheets['NA']['Device Type'] = sheets['NA']['Device Type'].replace(to_replace=r'.*Switch.*', value='Switch', regex=True)

    # Add the filtered DataFrames to the new dictionary
    filtered_sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name, df in sheets.items():
        filtered_sheets[sheet_name] = df

    return filtered_sheets


def get_device_dataset(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a combined device dataset from the filtered sheets.

    Parameters
    ----------
    sheets : Dict[str, pd.DataFrame]
        Dictionary where keys are worksheet (tab) names and values are
        the corresponding DataFrames.

    Returns
    -------
    pd.DataFrame
        Combined device dataset.
    """
    required_sheets = {'CatCtr', 'PrimeAP', 'PrimeWLC', 'NA', 'SOLID', 'SOLID-Loc', 'ModelData', 'Pricing'}
    missing_sheets = sorted(required_sheets - set(sheets))
    if missing_sheets:
        raise ValueError(
            f"The provided sheets dictionary is missing required sheets: {missing_sheets}"
        )

    def _normalize(series: pd.Series, strip_domain: bool = False) -> pd.Series:
        normalized = series.astype('string').str.strip().str.upper()
        if strip_domain:
            normalized = normalized.str.split('.').str[0]
        normalized = normalized.replace('', pd.NA)
        return normalized

    def _to_snake(label: str) -> str:
        cleaned = re.sub(r'[^0-9a-zA-Z]+', '_', str(label).strip().lower()).strip('_')
        return cleaned or 'col'

    def _map_type_from_text(raw_type: pd.Series) -> pd.Series:
        text = _normalize(raw_type, strip_domain=False)
        mapped = pd.Series(pd.NA, index=text.index, dtype='string')
        mapped = mapped.mask(text.str.contains(r'VOICE\s*GATEWAY', regex=True, na=False), 'Voice Gateway')
        mapped = mapped.mask(text.str.contains(r'WIRELESS\s*CONTROLLER|\bWLC\b', regex=True, na=False), 'Wireless Controller')
        mapped = mapped.mask(
            text.str.contains(r'ACCESS\s*POINT|UNIFIED\s*ACCESS\s*POINT|\bAP\b', regex=True, na=False),
            'AP',
        )
        mapped = mapped.mask(
            text.str.contains(r'ROUTER|INTEGRATED\s*SERVICES|EDGE\s*PLATFORM|\bASR\b', regex=True, na=False),
            'Router',
        )
        mapped = mapped.mask(text.str.contains(r'SWITCH|HUB', regex=True, na=False), 'Switch')
        return mapped

    def _build_device_frame(
        source_df: pd.DataFrame,
        source_name: str,
        id_col: str,
        model_col: str,
        extra_col_map: Dict[str, str],
        strip_domain_id: bool = True,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=source_df.index)
        frame['device_source'] = source_name
        frame['source_device_id'] = source_df[id_col].astype('string')
        frame['device_name'] = _normalize(source_df[id_col], strip_domain=strip_domain_id)
        frame['device_model_raw'] = _normalize(source_df[model_col], strip_domain=False)
        # Some platform/model fields contain stacked models delimited by commas.
        frame['device_model'] = frame['device_model_raw'].str.split(',').str[0].str.strip()
        frame['source_row_index'] = source_df.index

        for src_col, dst_col in extra_col_map.items():
            if src_col in source_df.columns:
                frame[dst_col] = source_df[src_col]

        return frame

    catctr = _build_device_frame(
        source_df=sheets['CatCtr'].copy(),
        source_name='CatCtr',
        id_col='hostname',
        model_col='platformId',
        extra_col_map={
            'dnsResolvedManagementAddress': 'device_ip',
            'family': 'device_family_raw',
            'platformId': 'source_platform_id',
            'reachabilityStatus': 'device_reachability',
            'serialNumber': 'serial_number',
            'softwareVersion': 'software_version',
            'type': 'device_type_raw',
            'vendor': 'device_vendor',
            'associatedWlcIp': 'associated_wlc_ip',
            'managementState': 'management_state',
            'role': 'device_role',
        },
        strip_domain_id=True,
    )

    prime_ap = _build_device_frame(
        source_df=sheets['PrimeAP'].copy(),
        source_name='PrimeAP',
        id_col='name',
        model_col='model',
        extra_col_map={
            'ipAddress': 'device_ip',
            'status': 'device_status',
            'serialNumber': 'serial_number',
            'softwareVersion': 'software_version',
            'type': 'device_type_raw',
            'controllerIpAddress': 'associated_wlc_ip',
            'controllerName': 'controller_name',
            'upTime': 'uptime',
            'countryCode': 'country_code',
        },
        strip_domain_id=True,
    )

    prime_wlc = _build_device_frame(
        source_df=sheets['PrimeWLC'].copy(),
        source_name='PrimeWLC',
        id_col='deviceName',
        model_col='manufacturer_part_partNumber',
        extra_col_map={
            'ipAddress': 'device_ip',
            'reachability': 'device_reachability',
            'manufacturer_part_serialNumber': 'serial_number',
            'softwareVersion': 'software_version',
            'softwareType': 'software_type',
            'deviceType': 'device_type_raw',
            'productFamily': 'device_family_raw',
            'adminStatus': 'device_status',
            'location': 'source_location',
        },
        strip_domain_id=True,
    )

    na = _build_device_frame(
        source_df=sheets['NA'].copy(),
        source_name='NA',
        id_col='Host Name',
        model_col='Device Model',
        extra_col_map={
            'Device IP': 'device_ip',
            'Device Type': 'device_type_raw',
            'Device Status': 'device_status',
            'Device Vendor': 'device_vendor',
            'Serial Number': 'serial_number',
            'Software Version': 'software_version',
            'Firmware Version': 'firmware_version',
            'Uptime': 'uptime',
            'Free Ports': 'free_ports',
            'Total Ports': 'total_ports',
            'Ports In Use': 'ports_in_use',
        },
        strip_domain_id=True,
    )

    devices = pd.concat([catctr, prime_ap, prime_wlc, na], ignore_index=True, sort=False)
    devices = devices[devices['device_name'].notna()].copy()

    # Canonicalize to one analysis-ready device type column.
    catctr_family = _normalize(devices.get('device_family_raw', pd.Series(pd.NA, index=devices.index)), strip_domain=False)
    catctr_type_from_family = catctr_family.replace(
        {
            'AP': 'AP',
            'ROUTER': 'Router',
            'SWITCH': 'Switch',
            'WIRELESS CONTROLLER': 'Wireless Controller',
            'VOICE GATEWAY': 'Voice Gateway',
        }
    ).astype('string')
    allowed_device_types = {'Router', 'Switch', 'Voice Gateway', 'AP', 'Wireless Controller'}
    catctr_type_from_family = catctr_type_from_family.where(catctr_type_from_family.isin(allowed_device_types), pd.NA)
    catctr_type_from_text = _map_type_from_text(devices.get('device_type_raw', pd.Series(pd.NA, index=devices.index)))
    catctr_device_type = catctr_type_from_family.fillna(catctr_type_from_text)

    na_device_type = _map_type_from_text(devices.get('device_type_raw', pd.Series(pd.NA, index=devices.index)))
    source_name = devices['device_source'].astype('string')
    devices['device_type'] = pd.Series(pd.NA, index=devices.index, dtype='string')
    devices.loc[source_name == 'CatCtr', 'device_type'] = catctr_device_type[source_name == 'CatCtr']
    devices.loc[source_name == 'PrimeAP', 'device_type'] = 'AP'
    devices.loc[source_name == 'PrimeWLC', 'device_type'] = 'Wireless Controller'
    devices.loc[source_name == 'NA', 'device_type'] = na_device_type[source_name == 'NA']

    source_priority = {'CatCtr': 1, 'PrimeAP': 2, 'PrimeWLC': 2, 'NA': 3}
    source_tiebreak = {'CatCtr': 1, 'PrimeAP': 2, 'PrimeWLC': 3, 'NA': 4}
    devices['source_priority'] = devices['device_source'].map(source_priority).fillna(999)
    devices['source_tiebreak'] = devices['device_source'].map(source_tiebreak).fillna(999)
    devices = devices.sort_values(
        by=['source_priority', 'device_name', 'source_tiebreak', 'source_row_index'],
        ascending=[True, True, True, True],
        kind='mergesort',
    )
    devices = devices.drop_duplicates(subset=['device_name'], keep='first').copy()

    devices['state_code'] = devices['device_name'].str[:2]
    devices['site_code'] = devices['device_name'].str[2:5]

    solid = sheets['SOLID'].copy()
    solid_loc = sheets['SOLID-Loc'].copy()

    solid['loc_site_code'] = _normalize(solid['Site Code'])
    solid['loc_state'] = _normalize(solid['State'])
    solid['loc_site_name'] = solid['Site Name']
    solid['loc_address_1'] = solid['Street Address 1']
    solid['loc_address_2'] = solid['Street Address 2']
    solid['loc_city'] = solid['City']
    solid['loc_zip'] = solid['Zip']

    solid_loc['loc_site_code'] = _normalize(solid_loc['Site Code'])
    solid_loc['loc_site_name_solid_loc'] = solid_loc['Site Name']
    solid_loc['loc_latitude'] = pd.to_numeric(solid_loc['Latitude'], errors='coerce')
    solid_loc['loc_longitude'] = pd.to_numeric(solid_loc['Longitude'], errors='coerce')
    solid_loc['loc_county'] = solid_loc['PhysicalAddressCounty']
    solid_loc['loc_call_group'] = solid_loc['Call Group']
    solid_loc['loc_owner'] = solid_loc['Owner']

    location = solid[
        [
            'loc_site_code',
            'loc_state',
            'loc_site_name',
            'loc_address_1',
            'loc_address_2',
            'loc_city',
            'loc_zip',
        ]
    ].merge(
        solid_loc[
            [
                'loc_site_code',
                'loc_site_name_solid_loc',
                'loc_latitude',
                'loc_longitude',
                'loc_county',
                'loc_call_group',
                'loc_owner',
            ]
        ],
        on='loc_site_code',
        how='left',
    )
    location['loc_site_name'] = location['loc_site_name'].fillna(location['loc_site_name_solid_loc'])
    location = location.drop(columns=['loc_site_name_solid_loc'])

    devices = devices.merge(
        location,
        left_on=['state_code', 'site_code'],
        right_on=['loc_state', 'loc_site_code'],
        how='left',
    )
    devices['location_matched'] = devices['loc_site_name'].notna()

    model_data = sheets['ModelData'].copy()
    url_cols = [col for col in model_data.columns if 'url' in str(col).lower()]
    model_data = model_data.drop(columns=url_cols, errors='ignore')

    model_data['model_key'] = _normalize(model_data['Model'])
    model_data['repl_device_key'] = _normalize(model_data['Repl Device'])

    model_rename_map = {
        col: f"modeldata_{_to_snake(col)}"
        for col in model_data.columns
        if col not in {'model_key', 'repl_device_key'}
    }
    model_data = model_data.rename(columns=model_rename_map)

    numeric_model_cols = [col for col in model_data.columns if col.endswith('_cost') or col.endswith('_hrs')]
    for col in numeric_model_cols:
        model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

    devices = devices.merge(
        model_data,
        left_on='device_model',
        right_on='model_key',
        how='left',
    )

    pricing = sheets['Pricing'].copy()
    pricing['parent_product_key'] = _normalize(pricing['Parent Product'])
    pricing['product_key'] = _normalize(pricing['Product'])
    pricing['device_flag'] = _normalize(pricing['Device?'])
    pricing['pricing_amount'] = pd.to_numeric(pricing['Pricing'], errors='coerce')
    pricing['labor_de_hours'] = pd.to_numeric(pricing['Labor-DE'], errors='coerce')
    pricing['labor_se_hours'] = pd.to_numeric(pricing['Labor-SE'], errors='coerce')
    pricing['labor_fo_hours'] = pd.to_numeric(pricing['Labor-FO'], errors='coerce')

    pricing_agg = pricing.groupby('parent_product_key', dropna=True, as_index=False).agg(
        pricing_component_count=('product_key', 'count'),
        pricing_total_estimate=('pricing_amount', 'sum'),
    )

    pricing_device_rows = pricing[pricing['device_flag'] == 'Y'][
        [
            'parent_product_key',
            'product_key',
            'pricing_amount',
            'labor_de_hours',
            'labor_se_hours',
            'labor_fo_hours',
        ]
    ].copy()
    pricing_device_rows = pricing_device_rows.rename(
        columns={
            'product_key': 'replacement_device_sku',
            'pricing_amount': 'replacement_device_cost',
            'labor_de_hours': 'replacement_labor_de_hours',
            'labor_se_hours': 'replacement_labor_se_hours',
            'labor_fo_hours': 'replacement_labor_fo_hours',
        }
    )

    pricing_features = pricing_agg.merge(pricing_device_rows, on='parent_product_key', how='left')

    devices = devices.merge(
        pricing_features,
        left_on='repl_device_key',
        right_on='parent_product_key',
        how='left',
    )

    helper_cols_to_drop = [
        'source_priority',
        'source_tiebreak',
        'source_row_index',
        'model_key',
        'parent_product_key',
        'device_type_raw',
        'device_family_raw',
    ]
    devices = devices.drop(columns=[col for col in helper_cols_to_drop if col in devices.columns])

    preferred_order = [
        'device_name',
        'device_source',
        'source_device_id',
        'state_code',
        'site_code',
        'location_matched',
        'loc_state',
        'loc_site_code',
        'loc_site_name',
        'loc_address_1',
        'loc_address_2',
        'loc_city',
        'loc_zip',
        'loc_county',
        'loc_call_group',
        'loc_owner',
        'loc_latitude',
        'loc_longitude',
        'device_model_raw',
        'device_model',
        'device_type',
        'device_status',
        'device_reachability',
        'device_vendor',
        'device_ip',
        'serial_number',
        'software_version',
        'firmware_version',
        'uptime',
        'controller_name',
        'associated_wlc_ip',
        'source_platform_id',
        'modeldata_model',
        'modeldata_model_parent',
        'modeldata_category',
        'modeldata_in_scope',
        'modeldata_repl_device',
        'modeldata_eos',
        'modeldata_eol',
        'modeldata_dna_y_n',
        'modeldata_dna_part_number',
        'modeldata_stg_config_y_n',
        'modeldata_de_hrs',
        'modeldata_se_hrs',
        'modeldata_fot_hrs',
        'modeldata_pm_hrs',
        'modeldata_de_cost',
        'modeldata_se_cost',
        'modeldata_fot_cost',
        'modeldata_labor_cost',
        'modeldata_device_cost',
        'modeldata_dna_cost',
        'modeldata_staging_cost',
        'modeldata_tax_oh',
        'modeldata_material_cost',
        'repl_device_key',
        'replacement_device_sku',
        'replacement_device_cost',
        'replacement_labor_de_hours',
        'replacement_labor_se_hours',
        'replacement_labor_fo_hours',
        'pricing_component_count',
        'pricing_total_estimate',
    ]

    ordered_columns = [col for col in preferred_order if col in devices.columns]
    remaining_columns = [col for col in devices.columns if col not in ordered_columns]

    return devices[ordered_columns + remaining_columns].reset_index(drop=True)


def clean_device_dataset(device_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the device dataset by removing duplicates and handling missing values.

    Parameters
    ----------
    device_dataset : pd.DataFrame
        The combined device dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned device dataset.
    """
    # Drop columns with significant missing values
    device_dataset.drop(
        columns=[
            'modeldata_pm_hrs',
            'software_type',
            'source_location',
            'country_code',
            'controller_name',
            'loc_address_2'
        ],
        inplace=True
    )

    # Drop columns that are not relevant for analysis
    device_dataset.drop(
        columns=[
            'device_model_raw',
            'firmware_version',
            'associated_wlc_ip',
            'source_platform_id',
            'management_state',
        ],
        inplace=True
    )

    return device_dataset
