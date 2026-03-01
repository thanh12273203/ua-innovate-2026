const API = {
  clusters: '/api/clusters',
  summary: '/api/location-summary',
  findings: '/api/findings',
};

const HORIZON_DAYS = 365;
const HOVER_DELAY_MS = 120;
const STATE_REFERENCE_ZOOM = 7;
const MIN_RADIUS_ZOOM_SCALE = 0.38;
const MAX_RADIUS_ZOOM_SCALE = 1.55;

const US_STATES_GEOJSON_URL =
  'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json';

const STATE_NAME_TO_ABBR = {
  Alabama: 'AL',
  Alaska: 'AK',
  Arizona: 'AZ',
  Arkansas: 'AR',
  California: 'CA',
  Colorado: 'CO',
  Connecticut: 'CT',
  Delaware: 'DE',
  Florida: 'FL',
  Georgia: 'GA',
  Hawaii: 'HI',
  Idaho: 'ID',
  Illinois: 'IL',
  Indiana: 'IN',
  Iowa: 'IA',
  Kansas: 'KS',
  Kentucky: 'KY',
  Louisiana: 'LA',
  Maine: 'ME',
  Maryland: 'MD',
  Massachusetts: 'MA',
  Michigan: 'MI',
  Minnesota: 'MN',
  Mississippi: 'MS',
  Missouri: 'MO',
  Montana: 'MT',
  Nebraska: 'NE',
  Nevada: 'NV',
  'New Hampshire': 'NH',
  'New Jersey': 'NJ',
  'New Mexico': 'NM',
  'New York': 'NY',
  'North Carolina': 'NC',
  'North Dakota': 'ND',
  Ohio: 'OH',
  Oklahoma: 'OK',
  Oregon: 'OR',
  Pennsylvania: 'PA',
  'Rhode Island': 'RI',
  'South Carolina': 'SC',
  'South Dakota': 'SD',
  Tennessee: 'TN',
  Texas: 'TX',
  Utah: 'UT',
  Vermont: 'VT',
  Virginia: 'VA',
  Washington: 'WA',
  'West Virginia': 'WV',
  Wisconsin: 'WI',
  Wyoming: 'WY',
  'District Of Columbia': 'DC',
};

const US_BOUNDS = L.latLngBounds(
  L.latLng(24.396308, -125.0),
  L.latLng(49.384358, -66.93457),
);

const map = L.map('us-map', {
  zoomControl: true,
  minZoom: 3,
  maxZoom: 12,
  maxBounds: US_BOUNDS.pad(0.2),
  maxBoundsViscosity: 0.8,
});
map.fitBounds(US_BOUNDS);

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
}).addTo(map);

let activeState = null;
let stateLayer = null;
const stateLabelLayer = L.layerGroup().addTo(map);
const clusterLayer = L.layerGroup().addTo(map);
let metricDomain = [-365, 365];
let maxDeviceCount = 1;
let globalMaxDeviceCount = 1;
let hoverTimer = null;
let hoverRequestToken = 0;
let highlightedMarker = null;
let highlightedMarkerBaseStyle = null;
let findingsLoaded = false;

const scopeTitle = document.getElementById('scope-title');
const scopeSubtitle = document.getElementById('scope-subtitle');
const resetButton = document.getElementById('reset-map-btn');
const summaryContent = document.getElementById('summary-content');
const findingsNote = document.getElementById('findings-note');
const findingsKpis = document.getElementById('findings-kpis');

function setupTabs() {
  const buttons = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      const targetId = button.dataset.tab;
      buttons.forEach((b) => b.classList.toggle('active', b === button));
      panels.forEach((panel) => panel.classList.toggle('active', panel.id === targetId));
      if (targetId === 'map-tab') {
        setTimeout(() => map.invalidateSize(), 100);
      }
      if (targetId === 'findings-tab' && !findingsLoaded) {
        loadFindings().catch((error) => {
          if (findingsNote) {
            findingsNote.textContent = `Unable to load findings: ${error.message}`;
          }
        });
      }
    });
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

function getStateAbbr(feature) {
  const name = feature?.properties?.name || '';
  return STATE_NAME_TO_ABBR[name] || '';
}

function toColor(value) {
  if (value === null || Number.isNaN(value)) {
    return '#8a959f';
  }
  const [rawMin, rawMax] = metricDomain;
  const min = Number.isFinite(rawMin) ? rawMin : -365;
  const max = Number.isFinite(rawMax) ? rawMax : 365;
  const safeMax = max === min ? max + 1 : max;
  const clamped = Math.max(min, Math.min(safeMax, value));
  const t = (clamped - min) / (safeMax - min);

  if (t < 0.5) {
    const k = t / 0.5;
    const r = 209;
    const g = Math.round(73 + (193 - 73) * k);
    const b = Math.round(91 + (78 - 91) * k);
    return `rgb(${r},${g},${b})`;
  }

  const k = (t - 0.5) / 0.5;
  const r = Math.round(242 + (42 - 242) * k);
  const g = Math.round(193 + (157 - 193) * k);
  const b = Math.round(78 + (80 - 78) * k);
  return `rgb(${r},${g},${b})`;
}

function toRadius(deviceCount) {
  const count = Number(deviceCount || 1);
  const scopeMaxDeviceCount = activeState ? maxDeviceCount : globalMaxDeviceCount;
  const maxRef = Math.max(1, Number(scopeMaxDeviceCount || 1));
  const baseRadius = 4 + (28 * Math.sqrt(count)) / Math.sqrt(maxRef);
  const zoomScale = Math.min(
    MAX_RADIUS_ZOOM_SCALE,
    Math.max(MIN_RADIUS_ZOOM_SCALE, Math.pow(2, (map.getZoom() - STATE_REFERENCE_ZOOM) * 0.6)),
  );
  return baseRadius * zoomScale;
}

function formatNumber(value, fractionDigits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: fractionDigits,
    minimumFractionDigits: fractionDigits,
  });
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderEolDistributionStackedBar(summary) {
  const horizonDays = Number(summary.horizon_days || HORIZON_DAYS);
  const total = Number(summary.total_devices ?? summary.device_count ?? 0);
  const overdue = Number(summary.overdue_count || 0);
  const within = Number(summary.within_horizon_count || 0);
  const future = Number(summary.future_count || 0);
  const unknown = Number(summary.unknown_count || 0);

  if (total <= 0) {
    return `
      <div class="mini-viz-block">
        <div class="mini-viz-title">EoL distribution (H=${horizonDays}d)</div>
        <div class="stacked-placeholder-bar" aria-hidden="true"></div>
        <div class="mini-viz-empty">No devices in location.</div>
      </div>
    `;
  }

  const segmentPercent = (count) => ((Number(count || 0) / total) * 100);

  return `
    <div class="mini-viz-block">
      <div class="mini-viz-title">EoL distribution (H=${horizonDays}d)</div>
      <div class="stacked-track" aria-label="100% stacked EoL distribution bar">
        <div class="segment seg-overdue" style="width:${segmentPercent(overdue)}%"></div>
        <div class="segment seg-within" style="width:${segmentPercent(within)}%"></div>
        <div class="segment seg-future" style="width:${segmentPercent(future)}%"></div>
        <div class="segment seg-unknown" style="width:${segmentPercent(unknown)}%"></div>
      </div>
      <div class="mini-legend">
        <span><i class="chip overdue"></i>Overdue</span>
        <span><i class="chip within"></i>Within</span>
        <span><i class="chip future"></i>Future</span>
        <span><i class="chip unknown"></i>Unknown</span>
      </div>
      <div class="mini-viz-meta">Overdue: ${formatNumber(overdue)} | Within ${horizonDays}d: ${formatNumber(within)} | Future: ${formatNumber(future)} | Unknown: ${formatNumber(unknown)}</div>
    </div>
  `;
}

function renderSummary(summary) {
  if (!summary || summary.found === false) {
    summaryContent.innerHTML = 'No summary data found for this location.';
    return;
  }

  const sourceEntries = Object.entries(summary.sources || {});
  const sourceText = sourceEntries.length
    ? sourceEntries.map(([k, v]) => `${k}: ${v}`).join(', ')
    : 'N/A';

  const stackedBarMiniViz = renderEolDistributionStackedBar(summary);

  summaryContent.innerHTML = `
    <div class="summary-grid">
      <div class="label">Location</div><div class="value">${escapeHtml(summary.site_name || 'N/A')}</div>
      <div class="label">Site Code</div><div class="value">${escapeHtml(summary.site_code || 'N/A')}</div>
      <div class="label">Devices</div><div class="value">${formatNumber(summary.device_count)}</div>
      <div class="label">EoL Known</div><div class="value">${formatNumber(summary.eol_known_devices)}</div>
      <div class="label">Overdue</div><div class="value">${formatNumber(summary.overdue_devices)}</div>
      <div class="label">Avg Remaining Days</div><div class="value">${formatNumber(summary.remaining_days_avg, 1)}</div>
    </div>
    ${stackedBarMiniViz}
    <p><strong>Sources:</strong> ${escapeHtml(sourceText)}</p>
  `;
}

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  return Number(value).toLocaleString(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  });
}

function setVizEmpty(containerId, message) {
  const el = document.getElementById(containerId);
  if (!el) {
    return;
  }
  el.innerHTML = `<div class="viz-empty">${escapeHtml(message)}</div>`;
}

function renderFindingsKpis(kpis) {
  if (!findingsKpis) {
    return;
  }
  findingsKpis.innerHTML = `
    <div class="kpi-card"><div class="kpi-label">Total Devices</div><div class="kpi-value">${formatNumber(kpis.total_devices)}</div></div>
    <div class="kpi-card"><div class="kpi-label">Total Sites</div><div class="kpi-value">${formatNumber(kpis.total_sites)}</div></div>
    <div class="kpi-card"><div class="kpi-label">Overdue Devices</div><div class="kpi-value">${formatNumber(kpis.overdue_devices)}</div></div>
    <div class="kpi-card"><div class="kpi-label">Unknown EoL Devices</div><div class="kpi-value">${formatNumber(kpis.unknown_eol_devices)}</div></div>
    <div class="kpi-card"><div class="kpi-label">Near-Term Cost Estimate</div><div class="kpi-value">${formatCurrency(kpis.near_term_cost_estimate)}</div></div>
  `;
}

function renderTopOverdueSites(topOverdueSites) {
  if (!window.Plotly) {
    setVizEmpty('viz-top-overdue', 'Plotly unavailable for this visualization.');
    return;
  }
  if (!topOverdueSites || topOverdueSites.length === 0) {
    setVizEmpty('viz-top-overdue', 'No overdue site data available.');
    return;
  }

  const rows = [...topOverdueSites].reverse();
  const y = rows.map((row) => `${row.state}-${row.site_code}`);
  const x = rows.map((row) => Number(row.overdue_count || 0));
  const text = rows.map(
    (row) => `${row.site_name || 'Unknown'}<br>Overdue: ${formatNumber(row.overdue_count)}<br>Near-Term Cost: ${formatCurrency(row.near_term_cost)}`,
  );

  Plotly.newPlot(
    'viz-top-overdue',
    [
      {
        type: 'bar',
        orientation: 'h',
        y,
        x,
        marker: { color: '#d1495b' },
        text,
        hovertemplate: '%{text}<extra></extra>',
      },
    ],
    {
      margin: { l: 70, r: 10, t: 10, b: 35 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      xaxis: { title: 'Overdue Device Count', gridcolor: '#e3ebe5' },
      yaxis: { automargin: true },
      height: 360,
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderRiskCostScatter(points) {
  if (!window.Plotly) {
    setVizEmpty('viz-risk-cost', 'Plotly unavailable for this visualization.');
    return;
  }
  if (!points || points.length === 0) {
    setVizEmpty('viz-risk-cost', 'No site-level risk/cost data available.');
    return;
  }

  const x = points.map((row) => Number(row.support_coverage_score || 0));
  const y = points.map((row) => Number(row.security_risk_score || 0));
  const cost = points.map((row) => Number(row.near_term_cost || 0));
  const overdueRate = points.map((row) => Number(row.overdue_rate || 0));
  const sizes = cost.map((v) => Math.max(7, Math.min(40, 7 + Math.sqrt(Math.max(0, v)) / 28)));
  const text = points.map(
    (row) =>
      `${row.site_name || 'Unknown'} (${row.state}-${row.site_code})<br>` +
      `Support Coverage: ${formatNumber(row.support_coverage_score, 1)}<br>` +
      `Security Risk: ${formatNumber(row.security_risk_score, 1)}<br>` +
      `Near-Term Cost: ${formatCurrency(row.near_term_cost)}<br>` +
      `Overdue Rate: ${formatNumber(row.overdue_rate, 1)}%`,
  );

  Plotly.newPlot(
    'viz-risk-cost',
    [
      {
        type: 'scatter',
        mode: 'markers',
        x,
        y,
        text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          size: sizes,
          color: overdueRate,
          colorscale: [
            [0.0, '#2a9d50'],
            [0.5, '#f2c14e'],
            [1.0, '#d1495b'],
          ],
          colorbar: { title: 'Overdue %' },
          line: { width: 0.7, color: '#243029' },
          opacity: 0.82,
        },
      },
    ],
    {
      margin: { l: 56, r: 24, t: 8, b: 42 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      height: 390,
      xaxis: { title: 'Support Coverage Score (higher is better)', range: [0, 100], gridcolor: '#e3ebe5' },
      yaxis: { title: 'Security Risk Proxy (higher is riskier)', range: [0, 100], gridcolor: '#e3ebe5' },
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderModelHotspots(modelRows) {
  if (!window.Plotly) {
    setVizEmpty('viz-model-hotspots', 'Plotly unavailable for this visualization.');
    return;
  }
  if (!modelRows || modelRows.length === 0) {
    setVizEmpty('viz-model-hotspots', 'No model hotspot data available.');
    return;
  }

  const rows = [...modelRows].reverse();
  Plotly.newPlot(
    'viz-model-hotspots',
    [
      {
        type: 'bar',
        orientation: 'h',
        y: rows.map((row) => row.device_model),
        x: rows.map((row) => Number(row.overdue_count || 0)),
        marker: { color: '#8d3fbc' },
        text: rows.map(
          (row) => `Devices: ${formatNumber(row.total_devices)} | Near-Term Cost: ${formatCurrency(row.near_term_cost)}`,
        ),
        hovertemplate:
          '%{y}<br>Overdue: %{x}<br>%{text}<br>Risk Score: %{customdata:.1f}<extra></extra>',
        customdata: rows.map((row) => Number(row.security_risk_score || 0)),
      },
    ],
    {
      margin: { l: 110, r: 10, t: 8, b: 35 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      height: 390,
      xaxis: { title: 'Overdue Device Count', gridcolor: '#e3ebe5' },
      yaxis: { automargin: true },
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderFindings(findings) {
  if (findingsNote) {
    findingsNote.textContent = findings.security_risk_note || 'Lifecycle analytics and risk-cost signals.';
  }
  renderFindingsKpis(findings.kpis || {});
  renderTopOverdueSites(findings.top_overdue_sites || []);
  renderRiskCostScatter(findings.site_risk_cost_scatter || []);
  renderModelHotspots(findings.model_hotspots || []);
}

async function loadFindings() {
  const query = new URLSearchParams({ horizon_days: String(HORIZON_DAYS) });
  const findings = await fetchJson(`${API.findings}?${query.toString()}`);
  renderFindings(findings);
  findingsLoaded = true;
}

function clearMarkerHighlight() {
  if (highlightedMarker && highlightedMarkerBaseStyle) {
    highlightedMarker.setStyle(highlightedMarkerBaseStyle);
  }
  highlightedMarker = null;
  highlightedMarkerBaseStyle = null;
}

function highlightMarker(marker) {
  const baseStyle = marker.options.__baseStyle;
  if (!baseStyle) {
    return;
  }
  clearMarkerHighlight();
  highlightedMarker = marker;
  highlightedMarkerBaseStyle = baseStyle;
  marker.setStyle({
    ...baseStyle,
    weight: 2.2,
    color: '#111',
    fillOpacity: 0.96,
  });
}

function setSummaryPlaceholder(message) {
  summaryContent.innerHTML = `<span>${escapeHtml(message)}</span>`;
}

async function fetchHoverSummary(cluster, marker) {
  const currentToken = ++hoverRequestToken;
  highlightMarker(marker);
  setSummaryPlaceholder(`Loading ${cluster.state}-${cluster.site_code}...`);

  try {
    const query = new URLSearchParams({
      state: cluster.state,
      site_code: cluster.site_code,
      horizon_days: String(HORIZON_DAYS),
    });
    const summary = await fetchJson(`${API.summary}?${query.toString()}`);
    if (currentToken !== hoverRequestToken) {
      return;
    }
    renderSummary(summary);
  } catch (error) {
    if (currentToken !== hoverRequestToken) {
      return;
    }
    setSummaryPlaceholder(`Unable to load summary: ${error.message}`);
  }
}

function buildMarkerBaseStyle(cluster) {
  return {
    radius: toRadius(cluster.device_count),
    color: '#223029',
    weight: 1.1,
    fillOpacity: 0.82,
    fillColor: toColor(cluster.remaining_days_weighted_total),
  };
}

function refreshClusterStyles() {
  clusterLayer.eachLayer((marker) => {
    const cluster = marker.options.__clusterData;
    if (!cluster) {
      return;
    }
    const baseStyle = buildMarkerBaseStyle(cluster);
    marker.options.__baseStyle = baseStyle;
    if (marker === highlightedMarker) {
      marker.setStyle({
        ...baseStyle,
        weight: 2.2,
        color: '#111',
        fillOpacity: 0.96,
      });
      highlightedMarkerBaseStyle = baseStyle;
    } else {
      marker.setStyle(baseStyle);
    }
  });
}

function renderClusters(clusters) {
  clusterLayer.clearLayers();
  clearMarkerHighlight();

  const orderedClusters = [...clusters].sort(
    (left, right) => Number(right.device_count || 0) - Number(left.device_count || 0),
  );

  orderedClusters.forEach((cluster) => {
    const baseStyle = buildMarkerBaseStyle(cluster);
    const marker = L.circleMarker([cluster.latitude, cluster.longitude], baseStyle);
    marker.options.__clusterData = cluster;
    marker.options.__baseStyle = baseStyle;
    marker.bindTooltip(
      `${cluster.site_name || 'Unknown Site'} (${cluster.state || ''}-${cluster.site_code || ''})`,
      { direction: 'top', opacity: 0.9, sticky: true },
    );

    marker.on('mouseover', () => {
      if (hoverTimer) {
        clearTimeout(hoverTimer);
      }
      hoverTimer = setTimeout(() => {
        fetchHoverSummary(cluster, marker);
      }, HOVER_DELAY_MS);
    });

    marker.on('mouseout', () => {
      if (hoverTimer) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
      }
      if (highlightedMarker === marker) {
        marker.setStyle(marker.options.__baseStyle || baseStyle);
        highlightedMarker = null;
        highlightedMarkerBaseStyle = null;
      }
    });

    marker.on('click', (event) => {
      if (event && event.originalEvent) {
        event.originalEvent.preventDefault();
        event.originalEvent.stopPropagation();
      }
      L.DomEvent.stop(event);
      if (document.activeElement && typeof document.activeElement.blur === 'function') {
        document.activeElement.blur();
      }
    });

    clusterLayer.addLayer(marker);
  });
}

function updateScopeText() {
  if (activeState) {
    scopeTitle.textContent = `State: ${activeState}`;
    scopeSubtitle.textContent = 'State-level view. Hover a cluster to inspect location summary.';
    resetButton.disabled = false;
  } else {
    scopeTitle.textContent = 'United States';
    scopeSubtitle.textContent = 'Click a state to zoom in. Hover clusters for location summary.';
    resetButton.disabled = true;
  }
}

async function loadClusters() {
  const query = activeState ? `?state=${encodeURIComponent(activeState)}` : '';
  const payload = await fetchJson(`${API.clusters}${query}`);
  metricDomain = payload.metric_domain || [-365, 365];
  maxDeviceCount = payload.max_device_count || 1;
  globalMaxDeviceCount = payload.global_max_device_count || payload.max_device_count || 1;
  renderClusters(payload.clusters || []);
}

async function selectState(state, bounds) {
  activeState = state;
  updateScopeText();
  if (bounds) {
    map.fitBounds(bounds.pad(0.04));
  }
  setSummaryPlaceholder('Hover a cluster to view summary statistics.');
  await loadClusters();
}

async function resetToUSView() {
  activeState = null;
  setSummaryPlaceholder('Hover a cluster to view location-level metrics.');
  updateScopeText();
  map.fitBounds(US_BOUNDS);
  await loadClusters();
}

function addStateLabels(features) {
  stateLabelLayer.clearLayers();
  features.forEach((feature) => {
    const abbr = getStateAbbr(feature);
    if (!abbr) {
      return;
    }
    const center = turf.centerOfMass(feature).geometry.coordinates;
    const label = L.marker([center[1], center[0]], {
      interactive: false,
      icon: L.divIcon({
        className: 'state-label',
        html: `<span>${abbr}</span>`,
      }),
    });
    stateLabelLayer.addLayer(label);
  });
}

async function loadStateLayer() {
  try {
    const geojson = await fetchJson(US_STATES_GEOJSON_URL);
    addStateLabels(geojson.features || []);

    stateLayer = L.geoJSON(geojson, {
      style: (feature) => {
        const abbr = getStateAbbr(feature);
        const selected = activeState && abbr === activeState;
        return {
          color: '#111',
          weight: selected ? 2.5 : 1.4,
          fillColor: selected ? '#9acfa8' : '#f4f8f2',
          fillOpacity: selected ? 0.24 : 0.05,
        };
      },
      onEachFeature: (feature, layer) => {
        const abbr = getStateAbbr(feature);
        if (!abbr) {
          return;
        }
        layer.on('click', async () => {
          if (activeState === abbr) {
            return;
          }
          await selectState(abbr, layer.getBounds());
          stateLayer.setStyle((featureItem) => {
            const candidate = getStateAbbr(featureItem);
            const selected = candidate === activeState;
            return {
              color: '#111',
              weight: selected ? 2.5 : 1.4,
              fillColor: selected ? '#9acfa8' : '#f4f8f2',
              fillOpacity: selected ? 0.24 : 0.05,
            };
          });
        });
      },
    }).addTo(map);
  } catch (error) {
    scopeSubtitle.textContent = `State boundary layer unavailable: ${error.message}`;
  }
}

resetButton.addEventListener('click', async () => {
  if (!activeState) {
    return;
  }
  await resetToUSView();
  if (stateLayer) {
    stateLayer.setStyle({
      color: '#111',
      weight: 1.4,
      fillColor: '#f4f8f2',
      fillOpacity: 0.05,
    });
  }
});

map.on('zoomend', () => {
  refreshClusterStyles();
});

async function initialize() {
  setupTabs();
  updateScopeText();
  setSummaryPlaceholder('Hover a cluster to view location-level metrics.');
  await Promise.all([loadStateLayer(), loadClusters()]);
}

initialize().catch((error) => {
  setSummaryPlaceholder(`Failed to initialize dashboard: ${error.message}`);
});
