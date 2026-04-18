"""
Streamlit A/B experiment analyzer: CSV upload, multi-metric roles, frequentist + Bayesian summaries.
"""
from __future__ import annotations

import hashlib
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import gaussian_filter1d
from scipy.stats import beta, gaussian_kde, ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# --- Constants ----------------------------------------------------------------

EXCLUDED_FROM_METRICS = frozenset(
    {"user_id", "variant", "date_in_experiment", "ab_name", "date_in_experivent"}
)
DEFAULT_CONTROL = "A"
DEFAULT_TREATMENT = "B"
BINARY_VALUES = {0, 1, 0.0, 1.0}
BOOTSTRAP_SAMPLES = 50_000
BAYES_SAMPLES = 50_000
UPLIFT_DIST_BOOTSTRAP = 2000  # user-level bootstrap for relative uplift viz

# Session keys — cached experiment dataframe & segmentation
EXP_DATA_FP_KEY = "_exp_data_fingerprint"
EXP_RESULTS_READY_KEY = "exp_results_ready"
EXP_BASE_DF_KEY = "exp_base_df"
EXP_WORK_DF_KEY = "exp_work_df"
EXP_ANALYSES_KEY = "exp_analyses_cache"
EXP_METRICS_SNAP_KEY = "exp_metrics_list"
EXP_CONTROL_SNAP_KEY = "exp_control_snap"
EXP_TREATMENT_SNAP_KEY = "exp_treatment_snap"
EXP_SEGMENT_REAPPLY_KEY = "_exp_segment_reapply"
EXP_RESULTS_VIEW_KEY = "exp_results_view"
RESULTS_VIEW_LABEL = "Results"
HEALTH_CHECK_VIEW_LABEL = "Health-check"

# Health-check thresholds (fractions of 1)
HC_SRM_THRESHOLD_PP = 0.05  # max deviation from 50/50 assignment (proportion points)
HC_COVARIATE_DIFF_THRESHOLD = 0.10  # highlight row if |pct_A − pct_B| exceeds this
HC_MAX_PROPERTY_CATEGORIES = 10
HC_VARIANT_COLOR_CTRL = "#60a5fa"  # subtle blue (arm A / control)
HC_VARIANT_COLOR_TRT = "#34d399"  # subtle green (arm B / treatment)

# Health-check covariate charts only (Altair): readability vs default Vega-Lite sizes
HC_CHART_LABEL_LIGHT = "#e8eaed"
HC_CHART_TITLE_LIGHT = "#f9fafb"
HC_CHART_X_LABEL_SIZE = 17  # category labels (~1.4× typical default)
HC_CHART_Y_LABEL_SIZE = 15  # tick % (~1.3×)
HC_CHART_AXIS_TITLE_SIZE = 17
HC_CHART_LEGEND_LABEL_SIZE = 15

SEGMENT_EXCLUDED_COLUMN_NAMES = frozenset(
    {"user_id", "variant", "date_in_experiment"}
)

# GrowthBook-style metrics panel (dark)
GB_TEXT_PRIMARY = "#FFFFFF"
GB_TEXT_SECONDARY = "#A0A0A0"
GB_POSITIVE = "#22C55E"
GB_NEGATIVE = "#EF4444"
GB_NEUTRAL = "#A0A0A0"
GB_ZERO_LINE = "#3B82F6"
GB_PANEL_BG = "#0F1115"
GB_CELL_LINE = "#272B36"
GB_DIST_BG = "#1A1D24"
GB_AXIS_LABEL = "#9CA3AF"
GB_AXIS_LABEL_STRONG = "#E5E7EB"
GB_AXIS_TICK_SUBTLE = "rgba(148,163,184,0.42)"
AXIS_LABEL_GAP_PX = 40.0
AXIS_LABEL_FONT_PX = 11
AXIS_MAX_LABELS = 8

APP_FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif'
)

GLOBAL_APP_STYLES = (
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
.stMarkdown, label, p, span:not(.rangeSlider), div, textarea, input {
  font-family: """
    + APP_FONT_STACK
    + """ !important;
}
[data-testid="stMarkdownContainer"] p, .stMarkdown p {
  line-height: 1.4 !important;
}
.block-container {
  line-height: 1.4 !important;
}
h1, h2, h3, h4, [data-testid="stHeader"] h1, [data-testid="stDecoration"] h1 {
  font-family: """
    + APP_FONT_STACK
    + """ !important;
  font-weight: 600 !important;
  font-style: normal !important;
  letter-spacing: -0.02em;
}

/* Hide Streamlit Material Symbol fallback text when the icon font fails (shows names like keyboard_double_arrow_right); show Unicode arrows instead. */
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button {
  position: relative;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="stExpandSidebarButton"] button > *,
[data-testid="stSidebarCollapseButton"] button > * {
  visibility: hidden !important;
  position: absolute !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
[data-testid="stExpandSidebarButton"] button::before {
  content: "→";
  display: inline-block;
  margin-right: 6px;
  opacity: 0.7;
  visibility: visible !important;
  font-size: 1.125rem;
  line-height: 1;
}
[data-testid="stSidebarCollapseButton"] button::before {
  content: "←";
  display: inline-block;
  margin-right: 6px;
  opacity: 0.7;
  visibility: visible !important;
  font-size: 1.125rem;
  line-height: 1;
}

/* App chrome: hide Material Symbol *names* leaked as span text (e.g. double_arrow_right); keep SVG if present */
[data-testid="stDecoration"] button:not([disabled]) span {
  display: inline-block !important;
  font-size: 0 !important;
  line-height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  position: absolute !important;
  pointer-events: none !important;
}
[data-testid="stDecoration"] button:not([disabled]) svg {
  display: block !important;
  visibility: visible !important;
  opacity: 0.9 !important;
}

[data-testid="stNavSectionHeader"] > div:last-child {
  position: relative;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-width: 1.125rem;
}
[data-testid="stNavSectionHeader"] > div:last-child > * {
  visibility: hidden !important;
  position: absolute !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
[data-testid="stNavSectionHeader"] > div:last-child::before {
  content: "▼";
  display: inline-block;
  margin-right: 6px;
  opacity: 0.7;
  visibility: visible !important;
  font-size: 0.875rem;
  line-height: 1;
}

/* Expander: hide Streamlit's built-in Material chevron when the font fails (raw names like keyboard_arrow_right overlap the label). No replacement icon — header stays a single-line label. */
[data-testid="stExpander"] summary > :first-child {
  display: none !important;
}

/* --- Role markers (zero-size; used only for CSS :has() scoping) --- */
.role-marker {
  display: block;
  width: 0;
  height: 0;
  margin: 0;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

/* Primary metric: single-select value (text only, no chip).
   IMPORTANT: Parent stVerticalBlocks can contain ALL role markers as descendants.
   Rules must match ONLY the inner block for that widget: one marker, no others. */
.primary-metric {
  color: #ff4d4f;
  font-weight: 700;
}
div[data-testid="stVerticalBlock"]:has(.role-marker--primary):not(:has(.role-marker--secondary)):not(:has(.role-marker--guardrail)):not(:has(.role-marker--property)) [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
  color: #ff4d4f !important;
  font-weight: 700 !important;
}

/* Secondary metrics multiselect (UI block only) → ORANGE chips */
div[data-testid="stVerticalBlock"]:has(.role-marker--secondary):not(:has(.role-marker--primary)):not(:has(.role-marker--guardrail)):not(:has(.role-marker--property)) [data-baseweb="tag"],
div[data-testid="stVerticalBlock"]:has(.role-marker--secondary):not(:has(.role-marker--primary)):not(:has(.role-marker--guardrail)):not(:has(.role-marker--property)) li[role="option"] {
  background-color: rgba(255, 149, 0, 0.2) !important;
  border: 1px solid rgba(255, 149, 0, 0.5) !important;
  color: #ff9500 !important;
  transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stVerticalBlock"]:has(.role-marker--secondary):not(:has(.role-marker--primary)):not(:has(.role-marker--guardrail)):not(:has(.role-marker--property)) [data-baseweb="tag"]:hover,
div[data-testid="stVerticalBlock"]:has(.role-marker--secondary):not(:has(.role-marker--primary)):not(:has(.role-marker--guardrail)):not(:has(.role-marker--property)) li[role="option"]:hover {
  background-color: rgba(255, 149, 0, 0.34) !important;
  cursor: pointer;
}

/* Guardrail metrics multiselect (UI block only) → BLUE chips */
div[data-testid="stVerticalBlock"]:has(.role-marker--guardrail):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--property)) [data-baseweb="tag"],
div[data-testid="stVerticalBlock"]:has(.role-marker--guardrail):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--property)) li[role="option"] {
  background-color: rgba(0, 122, 255, 0.2) !important;
  border: 1px solid rgba(0, 122, 255, 0.5) !important;
  color: #3ea6ff !important;
  transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stVerticalBlock"]:has(.role-marker--guardrail):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--property)) [data-baseweb="tag"]:hover,
div[data-testid="stVerticalBlock"]:has(.role-marker--guardrail):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--property)) li[role="option"]:hover {
  background-color: rgba(0, 122, 255, 0.34) !important;
  cursor: pointer;
}

/* Segment dimensions / user properties multiselect (UI block only) → GREEN chips */
div[data-testid="stVerticalBlock"]:has(.role-marker--property):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--guardrail)) [data-baseweb="tag"],
div[data-testid="stVerticalBlock"]:has(.role-marker--property):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--guardrail)) li[role="option"] {
  background-color: rgba(52, 199, 89, 0.2) !important;
  border: 1px solid rgba(52, 199, 89, 0.5) !important;
  color: #34c759 !important;
  transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stVerticalBlock"]:has(.role-marker--property):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--guardrail)) [data-baseweb="tag"]:hover,
div[data-testid="stVerticalBlock"]:has(.role-marker--property):not(:has(.role-marker--primary)):not(:has(.role-marker--secondary)):not(:has(.role-marker--guardrail)) li[role="option"]:hover {
  background-color: rgba(52, 199, 89, 0.34) !important;
  cursor: pointer;
}

.tag--secondary {
  display: inline-block;
  background: rgba(255, 149, 0, 0.2);
  color: #ff9500;
  border: 1px solid rgba(255, 149, 0, 0.5);
  border-radius: 6px;
  padding: 2px 8px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
.tag--secondary:hover {
  background: rgba(255, 149, 0, 0.34);
}
.tag--guardrail {
  display: inline-block;
  background: rgba(0, 122, 255, 0.2);
  color: #3ea6ff;
  border: 1px solid rgba(0, 122, 255, 0.5);
  border-radius: 6px;
  padding: 2px 8px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
.tag--guardrail:hover {
  background: rgba(0, 122, 255, 0.34);
}
.tag--property {
  display: inline-block;
  background: rgba(52, 199, 89, 0.2);
  color: #34c759;
  border: 1px solid rgba(52, 199, 89, 0.5);
  border-radius: 6px;
  padding: 2px 8px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
.tag--property:hover {
  background: rgba(52, 199, 89, 0.34);
}
.tag--other {
  display: inline-block;
  background: rgba(232, 121, 249, 0.22);
  color: #e879f9;
  border: 1px solid rgba(217, 70, 239, 0.45);
  border-radius: 6px;
  padding: 2px 8px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
.tag--other:hover {
  background: rgba(232, 121, 249, 0.36);
}

/* Results / Health-check segmented control (horizontal st.radio → Base Web radiogroup, pill style) */
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [data-testid="stRadio"] {
  margin-top: 10px !important;
  margin-bottom: 4px !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] {
  display: inline-flex !important;
  flex-direction: row !important;
  align-items: stretch !important;
  gap: 4px !important;
  padding: 4px !important;
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 12px !important;
  width: fit-content !important;
  max-width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] > [role="radio"] {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 8px 16px !important;
  margin: 0 !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  line-height: 1.25 !important;
  background: transparent !important;
  color: rgba(255, 255, 255, 0.52) !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  transition: background-color 0.18s ease, color 0.18s ease !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] > [role="radio"]:hover {
  background: rgba(255, 255, 255, 0.09) !important;
  color: rgba(255, 255, 255, 0.88) !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] > [role="radio"][aria-checked="true"] {
  background: #ff4d4f !important;
  color: #ffffff !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] > [role="radio"][aria-checked="true"]:hover {
  background: #ff6b6b !important;
  color: #ffffff !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [role="radiogroup"] > [role="radio"][aria-checked="true"] * {
  color: #ffffff !important;
}
/* Fallback: label + native radio */
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [data-testid="stRadio"] label {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 8px 16px !important;
  margin: 0 !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  background: transparent !important;
  color: rgba(255, 255, 255, 0.52) !important;
  border: none !important;
  transition: background-color 0.18s ease, color 0.18s ease !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [data-testid="stRadio"] label:hover {
  background: rgba(255, 255, 255, 0.09) !important;
  color: rgba(255, 255, 255, 0.85) !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [data-testid="stRadio"] label:has(input:checked) {
  background: #ff4d4f !important;
  color: #ffffff !important;
}
div[data-testid="stVerticalBlock"]:has(.results-view-segmented-wrap) [data-testid="stRadio"] label:has(input:checked):hover {
  background: #ff6b6b !important;
  color: #ffffff !important;
}

/* Health-check: variant A/B strip — match full width of Streamlit alert blocks below */
.variant-bar-container {
  display: flex;
  width: 100%;
  max-width: none;
  box-sizing: border-box;
}

/* Results → Goal metrics: variant split imbalance (filtered cohort) */
.goal-metrics-variant-imbalance-alert {
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.45);
  border-radius: 12px;
  padding: 14px 16px;
  margin: 0 0 16px 0;
  color: #fcd34d;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.45;
}
.goal-metrics-variant-imbalance-alert strong {
  color: #fde68a;
  font-weight: 600;
}
.goal-metrics-variant-imbalance-alert span {
  color: rgba(253, 230, 138, 0.92);
  font-weight: 500;
}

/* Single CSV upload zone: style native st.file_uploader only (no duplicate dashed markdown). */
div[data-testid="stVerticalBlock"]:has(.csv-upload-marker) [data-testid="stFileUploader"] {
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(.csv-upload-marker) [data-testid="stFileUploader"] > label {
  display: none !important;
}
div[data-testid="stVerticalBlock"]:has(.csv-upload-marker) [data-testid="stFileUploader"] section {
  border: 2px dashed #444 !important;
  border-radius: 12px !important;
  padding: 32px 20px !important;
  background: rgba(255, 255, 255, 0.03) !important;
  margin-bottom: 12px !important;
  text-align: center !important;
}
div[data-testid="stVerticalBlock"]:has(.csv-upload-marker) [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
div[data-testid="stVerticalBlock"]:has(.csv-upload-marker) [data-testid="stFileUploaderDropzone"] {
  border: none !important;
  background: transparent !important;
}
</style>
"""
)


# --- Small utilities ----------------------------------------------------------

def _norm_variant(s: Any) -> str:
    return str(s).strip()


def detect_metric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns except reserved experiment fields (case-insensitive names)."""
    excluded_lower = {c.lower() for c in EXCLUDED_FROM_METRICS}
    out: list[str] = []
    for c in df.columns:
        if c.lower() in excluded_lower:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def is_binary_metric(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = set(np.unique(s.astype(float).round(6)))
    return uniq.issubset(BINARY_VALUES)


def relative_uplift(control_mean: float, treatment_mean: float) -> float:
    if control_mean == 0:
        return float("nan") if treatment_mean != 0 else 0.0
    return (treatment_mean - control_mean) / abs(control_mean)


def prob_b_better_continuous(
    a: np.ndarray, b: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES
) -> float:
    """P(mean_B > mean_A) via bootstrap difference of means."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    mean_a = a[idx_a].mean(axis=1)
    mean_b = b[idx_b].mean(axis=1)
    return float(np.mean(mean_b > mean_a))


def prob_b_better_binary(conv_a: int, n_a: int, conv_b: int, n_b: int) -> float:
    posterior_a = beta.rvs(conv_a + 1, n_a - conv_a + 1, size=BAYES_SAMPLES)
    posterior_b = beta.rvs(conv_b + 1, n_b - conv_b + 1, size=BAYES_SAMPLES)
    return float(np.mean(posterior_b > posterior_a))


@dataclass
class MetricAnalysis:
    metric: str
    role: str
    kind: str  # "binary" | "continuous"
    n_control: int
    n_treatment: int
    value_control: float  # mean or conversion rate
    value_treatment: float
    uplift: float
    p_value: float
    probability: float  # P(treatment better) style


def extract_metric_arms(
    df: pd.DataFrame,
    metric: str,
    control_label: str,
    treatment_label: str,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    work = df[["variant", metric]].copy()
    work["variant"] = work["variant"].map(_norm_variant)
    c = work[work["variant"] == control_label][metric].dropna()
    t = work[work["variant"] == treatment_label][metric].dropna()
    if c.empty or t.empty:
        return None
    binary = is_binary_metric(work[metric])
    return c.astype(float).values, t.astype(float).values, binary


def bootstrap_uplift_samples(
    c_arm: np.ndarray,
    t_arm: np.ndarray,
    *,
    binary: bool = True,
    n_boot: int = UPLIFT_DIST_BOOTSTRAP,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap distribution of relative uplift ratios for **uncertainty visuals only**:
    KDE strip and percentile CI. Does **not** redefine the headline uplift.

    Each draw: resample users in A and B with replacement, mean_A/mean_B, uplift_i =
    (mean_B - mean_A) / mean_A (binary metrics use the same formula on bootstrap means).

    `binary` is accepted for call-site compatibility and ignored.
    """
    rng = np.random.default_rng(seed)
    c_arm = np.asarray(c_arm, dtype=float)
    t_arm = np.asarray(t_arm, dtype=float)
    n_a, n_b = len(c_arm), len(t_arm)
    if n_a < 1 or n_b < 1:
        return np.array([])

    idx_a = rng.integers(0, n_a, size=(n_boot, n_a))
    idx_b = rng.integers(0, n_b, size=(n_boot, n_b))
    pa = np.mean(c_arm[idx_a], axis=1)
    pb = np.mean(t_arm[idx_b], axis=1)

    denom = np.abs(pa)
    valid = denom > 1e-12
    uplift = np.full(n_boot, np.nan)
    uplift[valid] = (pb[valid] - pa[valid]) / denom[valid]
    return uplift[np.isfinite(uplift)]


def _eval_smoothed_density(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE on bootstrap draws; fallback to smoothed histogram."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    out = np.zeros_like(x_grid, dtype=float)
    if s.size < 5:
        return out

    sd = float(np.std(s))
    mn = float(np.mean(s))
    if sd < max(abs(mn), 1.0) * 1e-12:
        sigma = max(abs(mn) * 1e-5, 1e-12)
        out = np.exp(-0.5 * ((x_grid - mn) / sigma) ** 2)
        return np.maximum(out, 0.0)

    try:
        kde = gaussian_kde(s)
        out = kde(x_grid)
        out = np.maximum(out, 0.0)
    except (np.linalg.LinAlgError, ValueError):
        nb = min(56, max(16, s.size // 3))
        hist, edges = np.histogram(s, bins=nb, density=True)
        xc = (edges[:-1] + edges[1:]) / 2.0
        out = np.interp(x_grid, xc, hist, left=0.0, right=0.0)
        out = np.maximum(out, 0.0)

    # Extra spatial smoothing for a softer “hill”
    if len(out) >= 12:
        sigma_pix = max(1.4, len(out) / 90.0)
        out = gaussian_filter1d(out, sigma=sigma_pix, mode="nearest")
        out = np.maximum(out, 0.0)
    return out


def _squash_peak(y: np.ndarray, gamma: float = 0.58) -> np.ndarray:
    """Flatten peak / widen base (product-style density strip)."""
    y = np.maximum(y, 0.0)
    m = float(np.max(y))
    if m <= 0:
        return y
    return np.power(y / m, gamma)


def _svg_density_fill_path(x_px: np.ndarray, y_top: np.ndarray, baseline_y: float) -> str:
    """Closed path: baseline → curve → baseline (smooth outline, no jagged polygon tricks)."""
    x_px = np.asarray(x_px, dtype=float)
    y_top = np.asarray(y_top, dtype=float)
    parts: list[str] = [f"M {x_px[0]:.3f},{baseline_y:.3f}", f"L {x_px[0]:.3f},{y_top[0]:.3f}"]
    for i in range(1, len(x_px)):
        parts.append(f"L {x_px[i]:.3f},{y_top[i]:.3f}")
    parts.append(f"L {x_px[-1]:.3f},{baseline_y:.3f}")
    parts.append("Z")
    return " ".join(parts)


def _split_uplift_fill_paths(
    x_lin: np.ndarray,
    y_top: np.ndarray,
    baseline_y: float,
    x_px: np.ndarray,
    zero_px: float,
) -> tuple[str | None, str | None]:
    """
    Relative uplift x-axis: fill left of 0 red (loss), right green (gain).
    """
    xv_min = float(x_lin[0])
    xv_max = float(x_lin[-1])

    full = _svg_density_fill_path(x_px, y_top, baseline_y)

    if xv_max <= 0:
        return full, None
    if xv_min >= 0:
        return None, full

    y0 = float(np.interp(0.0, x_lin, y_top))
    iz = int(np.searchsorted(x_lin, 0.0, side="left"))
    iz = max(1, min(iz, len(x_lin) - 1))

    parts_l: list[str] = [
        f"M {x_px[0]:.3f},{baseline_y:.3f}",
        f"L {x_px[0]:.3f},{y_top[0]:.3f}",
    ]
    for i in range(1, iz):
        parts_l.append(f"L {x_px[i]:.3f},{y_top[i]:.3f}")
    parts_l.append(f"L {zero_px:.3f},{y0:.3f}")
    parts_l.append(f"L {zero_px:.3f},{baseline_y:.3f}")
    parts_l.append("Z")
    path_left = " ".join(parts_l)

    parts_r: list[str] = [
        f"M {zero_px:.3f},{baseline_y:.3f}",
        f"L {zero_px:.3f},{y0:.3f}",
    ]
    for i in range(iz, len(x_lin)):
        parts_r.append(f"L {x_px[i]:.3f},{y_top[i]:.3f}")
    parts_r.append(f"L {x_px[-1]:.3f},{baseline_y:.3f}")
    parts_r.append("Z")
    path_right = " ".join(parts_r)

    return path_left, path_right


GROWTHBOOK_METRICS_STYLES = """
<style>
.gb-metrics-shell {
  font-family: """ + APP_FONT_STACK + """;
  background: """ + GB_PANEL_BG + """;
  border-radius: 12px;
  padding: 18px 20px 12px 20px;
  margin: 8px 0 16px 0;
  overflow: visible;
}
.gb-table-wrapper {
  max-height: 70vh;
  overflow-y: auto;
  overflow-x: auto;
  position: relative;
}
.gb-table-header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: #0b0f17;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
}
.gb-table-body {
  position: relative;
}
.gb-metric--sticky-col {
  position: sticky;
  left: 0;
  z-index: 12;
  background: """ + GB_PANEL_BG + """;
  box-shadow: 4px 0 14px rgba(0,0,0,0.35);
}
.gb-table-header .gb-metric--sticky-col {
  z-index: 25;
  background: #0b0f17;
  box-shadow: 4px 0 14px rgba(0,0,0,0.45);
}
.gb-section-wrap.gb-section-secondary .gb-metric--sticky-col {
  background: rgba(15,17,21,0.98);
}
.gb-section-wrap.gb-section-guardrail .gb-metric--sticky-col {
  background: rgba(15,17,21,0.98);
}
.gb-section-wrap.gb-section-other .gb-metric--sticky-col {
  background: rgba(15,17,21,0.98);
}
.gb-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 16px;
  padding-bottom: 14px;
  border-bottom: 1px solid """ + GB_CELL_LINE + """;
  line-height: 1.4;
}
.gb-row:last-of-type { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.gb-header-row {
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #3F4654;
}
.gb-section-wrap {
  border-radius: 8px;
}
.gb-section-wrap.gb-section-secondary {
  background: rgba(255,255,255,0.02);
  padding: 10px 12px 6px;
  margin-left: -8px;
  margin-right: -8px;
}
.gb-section-wrap.gb-section-guardrail {
  background: rgba(255,255,255,0.012);
  padding: 10px 12px 6px;
  margin-left: -8px;
  margin-right: -8px;
}
.gb-section-wrap.gb-section-other {
  background: rgba(255,255,255,0.015);
  padding: 10px 12px 6px;
  margin-left: -8px;
  margin-right: -8px;
}
.gb-section-head {
  font-size: 13px;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 24px;
  margin-bottom: 8px;
}
.gb-section-head.gb-section-head-first {
  margin-top: 0;
}
.gb-section-divider {
  height: 1px;
  background: rgba(255,255,255,0.06);
  margin-bottom: 8px;
  width: 100%;
}
.gb-cell {
  flex: 1 1 0;
  min-width: 0;
}
.gb-metric { flex: 1.35 1 128px; }
.gb-base, .gb-var { flex: 1 1 96px; }
.gb-chance { flex: 0.95 1 88px; text-align: left; }
.gb-dist {
  flex: 2.4 1 300px;
  min-width: 280px;
  overflow: visible;
}
.gb-pct { flex: 0.85 1 76px; text-align: right; }
.gb-hdr {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: """ + GB_TEXT_SECONDARY + """;
}
.gb-metric-title {
  font-size: 15px;
  line-height: 1.4;
  font-family: """ + APP_FONT_STACK + """;
}
.gb-metric-title.primary-metric {
  color: #ff4d4f;
  font-weight: 700;
}
.gb-metric-title.tag--secondary {
  color: #ff9500;
  font-weight: 700;
}
.gb-metric-title.tag--guardrail {
  color: #3ea6ff;
  font-weight: 700;
}
.gb-metric-title.tag--property {
  color: #34c759;
  font-weight: 700;
}
.gb-metric-title.tag--other {
  color: #e879f9;
  font-weight: 700;
}
.dist-bar {
  position: relative;
  width: 100%;
  max-width: 320px;
  height: 40px;
  background: """ + GB_DIST_BG + """;
  border-radius: 6px;
  overflow: hidden;
}
.dist-bar .ci-band {
  position: absolute;
  top: 0;
  bottom: 0;
  background: rgba(255,255,255,0.09);
  border-left: 1px solid rgba(255,255,255,0.06);
  border-right: 1px solid rgba(255,255,255,0.06);
  z-index: 2;
  pointer-events: none;
}
.dist-bar .zero-line {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  margin-left: -1px;
  background: """ + GB_ZERO_LINE + """;
  z-index: 5;
  box-shadow: 0 0 6px rgba(59,130,246,0.35);
}
.dist-bar .mean-line {
  position: absolute;
  top: 2px;
  bottom: 2px;
  width: 1px;
  margin-left: -0.5px;
  border-left: 2px dashed rgba(255,255,255,0.85);
  z-index: 6;
  pointer-events: none;
}
.dist-bar .density-layer {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  z-index: 1;
}
.dist-bar .density-layer svg {
  display: block;
  width: 100%;
  height: 100%;
}
.gb-axis-row {
  align-items: flex-end;
  margin-top: 6px;
  margin-bottom: 10px;
  padding-top: 8px;
  padding-bottom: 10px;
  border-bottom: 1px solid """ + GB_CELL_LINE + """;
  min-height: 44px;
}
.gb-global-dist-axis {
  width: 100%;
  max-width: 320px;
  height: 38px;
  align-self: flex-start;
}
.gb-global-dist-axis svg {
  display: block;
  width: 100%;
  height: 100%;
}
.dist-wrap {
  position: relative;
  width: 100%;
  max-width: 320px;
  align-self: flex-start;
}
.dist-wrap:hover {
  z-index: 20;
}
.gb-tooltip,
.tooltip {
  position: absolute;
  top: -70px;
  left: 50%;
  transform: translateX(-50%);
  background: #111827;
  color: #E5E7EB;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.35;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s ease;
  white-space: nowrap;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  z-index: 10;
}
.dist-wrap:hover .gb-tooltip,
.dist-wrap:hover .tooltip {
  opacity: 1;
}

.segment-filters-card {
  background: rgba(255,255,255,0.035);
  border-radius: 14px;
  padding: 18px 20px;
  margin: 0 0 20px 0;
  border: 1px solid rgba(255,255,255,0.07);
}
.segment-filters-card h4 {
  margin: 0 0 12px 0;
  font-size: 1rem;
  font-weight: 600;
  color: #E5E7EB;
}
.segment-filter-label {
  color: #9CA3AF;
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin: 12px 0 6px 0;
}
.segment-filter-users {
  margin-top: 14px;
  font-size: 0.9rem;
  color: #D1D5DB;
}
</style>
"""


def chance_to_win_style(probability: float) -> str:
    if probability > 0.60:
        return GB_POSITIVE
    if probability < 0.40:
        return GB_NEGATIVE
    return GB_NEUTRAL


def primary_summary_chance_color(probability: float) -> str:
    """Goal-metrics summary: chance-to-win band colors (probability in [0, 1])."""
    if np.isnan(probability):
        return GB_NEUTRAL
    p = float(probability)
    if p >= 0.95:
        return "#059669"
    if p >= 0.70:
        return GB_POSITIVE
    if p >= 0.40:
        return GB_NEUTRAL
    return GB_NEGATIVE


def primary_uplift_summary_color(uplift: float) -> str:
    """KPI strip: uplift value color (positive / negative / flat)."""
    if np.isnan(uplift):
        return "#9CA3AF"
    if uplift > 0:
        return GB_POSITIVE
    if uplift < 0:
        return GB_NEGATIVE
    return GB_NEUTRAL


def build_goal_kpi_strip_fragment(
    ctrl: str,
    trt: str,
    n_c: int,
    n_t: int,
    primary: str | None,
    primary_ma: MetricAnalysis | None,
) -> str:
    """
    Self-contained HTML (+ CSS) for the Goal metrics KPI row.
    Rendered via components.html so flex layout is not broken by Streamlit markdown.
    """
    esc = html.escape
    if primary_ma is not None and not np.isnan(primary_ma.uplift):
        upl_raw = primary_ma.uplift
        upl_text = esc(f"{upl_raw:+.1%}")
        upl_color = primary_uplift_summary_color(float(upl_raw))
    else:
        upl_text = "—"
        upl_color = "rgba(255,255,255,0.45)"
    if primary_ma is not None and not np.isnan(primary_ma.probability):
        cw_text = esc(f"{primary_ma.probability * 100:.1f}%")
        cw_color = primary_summary_chance_color(float(primary_ma.probability))
    else:
        cw_text = "—"
        cw_color = "rgba(255,255,255,0.45)"
    metric_display = esc(str(primary)) if primary else "—"
    font = APP_FONT_STACK
    return f"""<style>
body {{ margin: 0; background: transparent; overflow: hidden; }}
.kpi-container {{
  padding: 20px;
  border-radius: 12px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  box-sizing: border-box;
  font-family: {font};
  margin-bottom: 16px;
}}
.kpi-row {{
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 40px;
  width: 100%;
  box-sizing: border-box;
}}
.kpi-item {{
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 120px;
  flex: 1 1 auto;
}}
.kpi-label {{
  font-size: 12px;
  font-weight: 500;
  color: rgba(255,255,255,0.5);
  line-height: 1.2;
}}
.kpi-value {{
  font-size: clamp(26px, 4vw, 34px);
  font-weight: 600;
  color: #ffffff;
  line-height: 1.15;
}}
.kpi-value.kpi-value--metric {{
  font-size: 16px;
  font-weight: 600;
  color: rgba(255,255,255,0.8);
}}
.kpi-value.kpi-value--metric.primary-metric {{
  color: #ff4d4f !important;
  font-weight: 700 !important;
}}
.kpi-value.kpi-value--uplift {{
  font-weight: 700;
}}
.kpi-value.kpi-value--chance {{
  font-size: clamp(24px, 3.6vw, 30px);
  font-weight: 600;
}}
</style>
<div class="kpi-container">
  <div class="kpi-row">
    <div class="kpi-item">
      <div class="kpi-label">{esc(f"Users ({ctrl})")}</div>
      <div class="kpi-value">{esc(f"{n_c:,}")}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">{esc(f"Users ({trt})")}</div>
      <div class="kpi-value">{esc(f"{n_t:,}")}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Primary metric</div>
      <div class="kpi-value kpi-value--metric primary-metric">{metric_display}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Primary uplift</div>
      <div class="kpi-value kpi-value--uplift" style="color:{upl_color};">{upl_text}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Chance to win</div>
      <div class="kpi-value kpi-value--chance" style="color:{cw_color};">{cw_text}</div>
    </div>
  </div>
</div>
"""


def _hc_prepare_property_series(s: pd.Series) -> pd.Series:
    out = s.copy()
    return out.astype(object).where(out.notna(), "(missing)")


def _hc_top_categories_other(s: pd.Series, max_cats: int) -> tuple[pd.Series, list[Any]]:
    """Bucket rare levels into \"Other\"; return series and category order (by frequency)."""
    s = _hc_prepare_property_series(s)
    vc = s.value_counts()
    if len(vc) <= max_cats:
        return s, list(vc.index)
    keep = set(vc.head(max_cats).index)
    mapped = s.where(s.isin(keep), "Other")
    ord_list = list(vc.head(max_cats).index)
    if (mapped == "Other").any():
        ord_list.append("Other")
    return mapped, ord_list


def compute_variant_balance_for_srm(
    df: pd.DataFrame, ctrl: str, trt: str
) -> dict[str, Any]:
    """Two-arm counts, observed split, and SRM flag (expected 50/50 by default)."""
    n_c = int((df["variant"] == ctrl).sum())
    n_t = int((df["variant"] == trt).sum())
    n = n_c + n_t
    if n == 0:
        return {
            "n_c": 0,
            "n_t": 0,
            "p_c": 0.0,
            "p_t": 0.0,
            "dev_pp": 0.0,
            "srm_ok": True,
        }
    p_c = n_c / n
    p_t = n_t / n
    dev_pp = max(abs(p_c - 0.5), abs(p_t - 0.5))
    srm_ok = dev_pp <= HC_SRM_THRESHOLD_PP
    return {
        "n_c": n_c,
        "n_t": n_t,
        "p_c": p_c,
        "p_t": p_t,
        "dev_pp": dev_pp,
        "srm_ok": srm_ok,
    }


def covariate_balance_property(
    df: pd.DataFrame, prop: str, ctrl: str, trt: str
) -> tuple[pd.DataFrame, int]:
    """
    Per category: % of ctrl users with value, % of trt users with value, absolute diff.
    Returns (table, number of rows exceeding HC_COVARIATE_DIFF_THRESHOLD).
    """
    if prop not in df.columns:
        return pd.DataFrame(), 0
    mapped, cat_order = _hc_top_categories_other(df[prop], HC_MAX_PROPERTY_CATEGORIES)
    sub = pd.DataFrame({"variant": df["variant"].values, "_cat": mapped.values})
    m_a = sub["variant"] == ctrl
    m_b = sub["variant"] == trt
    n_a = int(m_a.sum())
    n_b = int(m_b.sum())
    rows: list[dict[str, Any]] = []
    for cat in cat_order:
        if cat not in sub["_cat"].values:
            continue
        c_a = int((m_a & (sub["_cat"] == cat)).sum())
        c_b = int((m_b & (sub["_cat"] == cat)).sum())
        pct_a = c_a / n_a if n_a else 0.0
        pct_b = c_b / n_b if n_b else 0.0
        diff = abs(pct_a - pct_b)
        rows.append(
            {
                "value": cat,
                "pct_ctrl": pct_a,
                "pct_trt": pct_b,
                "diff": diff,
                "flag": diff > HC_COVARIATE_DIFF_THRESHOLD,
            }
        )
    out = pd.DataFrame(rows)
    n_flag = int(out["flag"].sum()) if not out.empty else 0
    return out, n_flag


def build_covariate_balance_chart(
    balance_df: pd.DataFrame,
    ctrl: str,
    trt: str,
    ctrl_label: str,
    trt_label: str,
) -> alt.Chart | None:
    """Grouped bars: x = category, two bars per category (pct of arm users)."""
    if balance_df.empty:
        return None
    plot = balance_df.rename(
        columns={"value": "Category", "pct_ctrl": ctrl_label, "pct_trt": trt_label}
    )
    long_df = plot.melt(
        id_vars=["Category"],
        value_vars=[ctrl_label, trt_label],
        var_name="Arm",
        value_name="pct",
    )
    sort_order = balance_df["value"].tolist()
    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Category:N",
                title=None,
                sort=sort_order,
                axis=alt.Axis(
                    labelAngle=-35,
                    labelLimit=200,
                    labelFontSize=HC_CHART_X_LABEL_SIZE,
                    labelColor=HC_CHART_LABEL_LIGHT,
                ),
            ),
            y=alt.Y(
                "pct:Q",
                title="% of users in arm",
                axis=alt.Axis(
                    format=".0%",
                    grid=True,
                    labelFontSize=HC_CHART_Y_LABEL_SIZE,
                    labelColor=HC_CHART_LABEL_LIGHT,
                    titleFontSize=HC_CHART_AXIS_TITLE_SIZE,
                    titleFontWeight="bold",
                    titleColor=HC_CHART_TITLE_LIGHT,
                ),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Arm:N",
                scale=alt.Scale(
                    domain=[ctrl_label, trt_label],
                    range=[HC_VARIANT_COLOR_CTRL, HC_VARIANT_COLOR_TRT],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelFontSize=HC_CHART_LEGEND_LABEL_SIZE,
                    labelColor=HC_CHART_LABEL_LIGHT,
                ),
            ),
            xOffset=alt.XOffset("Arm:N"),
            tooltip=[
                alt.Tooltip("Category:N", title="Value"),
                alt.Tooltip("Arm:N", title="Arm"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=560)
        .configure_axis(gridColor="rgba(255,255,255,0.08)")
        .configure_view(strokeWidth=0)
        .configure(font="Inter")
    )


def render_health_check_view(
    df: pd.DataFrame,
    ctrl: str,
    trt: str,
    sel_props: list[str],
) -> None:
    """Variant SRM strip + per-property covariate balance (uses filtered df only)."""
    esc = html.escape
    stats = compute_variant_balance_for_srm(df, ctrl, trt)
    total_imbalance_rows = 0
    prop_blocks: list[tuple[str, pd.DataFrame, int]] = []

    base_cols = [c for c in df.columns if c in sel_props]
    for prop in base_cols:
        bdf, n_bad = covariate_balance_property(df, prop, ctrl, trt)
        total_imbalance_rows += n_bad
        prop_blocks.append((prop, bdf, n_bad))

    # --- Section 1: variant split & SRM (assignment balance)
    st.subheader("Variant distribution")
    if stats["n_c"] + stats["n_t"] == 0:
        st.warning("No rows in filtered cohort.")
        return

    st.markdown(
        f"**{esc(str(ctrl))}:** {stats['n_c']:,} users ({stats['p_c']:.1%})  \n"
        f"**{esc(str(trt))}:** {stats['n_t']:,} users ({stats['p_t']:.1%})"
    )
    pw = stats["p_c"] * 100.0
    qw = stats["p_t"] * 100.0
    bar_html = (
        f'<div class="variant-bar-container" style="margin:12px 0 8px 0;height:36px;'
        f'border-radius:8px;overflow:hidden;border:1px solid rgba(148,163,184,0.35);">'
        f'<div style="width:{pw:.4f}%;min-width:0;background:{HC_VARIANT_COLOR_CTRL};'
        f'display:flex;align-items:center;justify-content:center;color:#0f172a;font-weight:700;font-size:13px;">'
        f"{esc(str(ctrl))}</div>"
        f'<div style="width:{qw:.4f}%;min-width:0;background:{HC_VARIANT_COLOR_TRT};'
        f'display:flex;align-items:center;justify-content:center;color:#0f172a;font-weight:700;font-size:13px;">'
        f"{esc(str(trt))}</div></div>"
    )
    st.markdown(bar_html, unsafe_allow_html=True)

    if stats["srm_ok"]:
        st.success("Distribution looks balanced (relative to 50/50 expected split).")
    else:
        st.warning(
            "Possible sample ratio mismatch (SRM): assignment deviates from 50/50 "
            f"by more than {HC_SRM_THRESHOLD_PP:.0%} (observed "
            f"{stats['p_c']:.1%} vs {stats['p_t']:.1%})."
        )

    if not base_cols:
        st.caption(
            "Select **Segment dimensions** in Step 2 to compare property mix between arms."
        )
        return

    st.divider()
    # --- Section 2: covariates (summary + per-property blocks)
    if total_imbalance_rows > 0:
        st.warning(
            f"**{total_imbalance_rows}** imbalance(s) detected across property categories "
            f"(|Δ| > {HC_COVARIATE_DIFF_THRESHOLD:.0%} between **{esc(str(ctrl))}** and "
            f"**{esc(str(trt))}** within a row)."
        )
    else:
        st.success(
            "No strong covariate imbalances flagged for the selected properties "
            f"(threshold |Δ| ≤ {HC_COVARIATE_DIFF_THRESHOLD:.0%})."
        )

    st.subheader("User property distributions")
    ctrl_pct_col = f"{ctrl} %"
    trt_pct_col = f"{trt} %"

    for prop, bdf, n_bad_prop in prop_blocks:
        if bdf.empty:
            continue
        st.markdown(f"#### Distribution by **{esc(str(prop))}**")
        if n_bad_prop:
            st.caption(f"⚠️ {n_bad_prop} category row(s) exceed imbalance threshold.")

        chart = build_covariate_balance_chart(
            bdf, ctrl, trt, ctrl_pct_col, trt_pct_col
        )
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

        disp = bdf.copy()
        disp[ctrl_pct_col] = disp["pct_ctrl"].map(lambda x: f"{x:.1%}")
        disp[trt_pct_col] = disp["pct_trt"].map(lambda x: f"{x:.1%}")
        disp["Diff"] = disp["diff"].map(lambda x: f"{x:.1%}")
        show = disp[["value", ctrl_pct_col, trt_pct_col, "Diff"]].rename(
            columns={"value": "Value"}
        )

        def _style_imbalance(row: pd.Series) -> list[str]:
            idx = row.name
            bad = bool(bdf.iloc[idx]["flag"]) if idx is not None else False
            if bad:
                return ["background-color: rgba(239,68,68,0.28)"] * len(row)
            return [""] * len(row)

        try:
            st.dataframe(
                show.style.apply(_style_imbalance, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("")


def format_arm_cell(ma: MetricAnalysis, arm: np.ndarray, *, is_control: bool) -> tuple[str, str]:
    """Primary value + secondary line (total / n)."""
    n = len(arm)
    if ma.kind == "binary":
        val = ma.value_control if is_control else ma.value_treatment
        conv = int(np.sum(arm))
        main = f"{val:.2%}"
        sub = f"{conv:,} / {n:,}"
    else:
        val = ma.value_control if is_control else ma.value_treatment
        total = float(np.sum(arm))
        main = f"{val:.4g}"
        sub = f"{total:,.4g} / {n:,}"
    return main, sub


def format_pct_change_html(uplift: float) -> str:
    if np.isnan(uplift):
        return f'<span style="color:{GB_TEXT_SECONDARY};font-weight:500;">—</span>'
    if uplift > 0:
        return (
            f'<span style="color:{GB_POSITIVE};font-weight:500;">↑ +{uplift * 100:.1f}%</span>'
        )
    if uplift < 0:
        return (
            f'<span style="color:{GB_NEGATIVE};font-weight:500;">↓ -{abs(uplift * 100):.1f}%</span>'
        )
    return f'<span style="color:{GB_TEXT_SECONDARY};font-weight:500;">0%</span>'


def compute_global_uplift_xlim(
    uplift_arrays: list[np.ndarray], *, pad_frac: float = 0.15
) -> tuple[float, float]:
    """
    Shared x-range for all distribution strips: padding on min/max of pooled samples; 0 always visible.
    """
    chunks = [
        np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
        for a in uplift_arrays
    ]
    chunks = [c for c in chunks if c.size > 0]
    if not chunks:
        return -0.5, 0.5
    u = np.concatenate(chunks)
    lo = float(np.min(u))
    hi = float(np.max(u))
    span = max(hi - lo, max(abs(lo), abs(hi)) * 1e-9, 1e-12)
    pad = span * pad_frac
    x_min = min(lo, 0.0) - pad
    x_max = max(hi, 0.0) + pad
    if x_max <= x_min:
        x_max = x_min + 1e-9
    return x_min, x_max


def _axis_minor_tick_step(x_min: float, x_max: float) -> float:
    """Dense tick marks every 5% or 10% based on span (thin stems only)."""
    span = x_max - x_min
    if span <= 1e-18:
        return 0.05
    if span <= 2.5:
        return 0.05
    return 0.10


def _enumerate_ticks(step: float, lo: float, hi: float) -> list[float]:
    out: list[float] = []
    t = math.ceil(lo / step - 1e-12) * step
    for _ in range(600):
        if t > hi + step * 1e-12:
            break
        if lo - 1e-12 <= t <= hi + 1e-12:
            out.append(round(t, 12))
        t += step
    return out


def _axis_minor_ticks(x_min: float, x_max: float) -> list[float]:
    """Many thin vertical stems on a regular %-grid."""
    span = x_max - x_min
    if span <= 1e-18:
        return [float(x_min)]
    step = _axis_minor_tick_step(x_min, x_max)
    ticks = _enumerate_ticks(step, x_min, x_max)
    cap = 48
    while len(ticks) > cap and step < 2.0:
        step *= 2.0
        ticks = _enumerate_ticks(step, x_min, x_max)
    if len(ticks) < 2:
        ticks = [float(x_min), float(x_max)]
    return ticks


def _axis_sparse_labels(
    x_min: float,
    x_max: float,
    width_px: float,
) -> list[float]:
    """
    Few text labels (≤8): sparse grid (prefers ~every 20%) plus required extrema and 0%.
    Dense minor ticks are drawn separately; labels skip collisions in pixel space.
    """
    span = x_max - x_min
    if span <= 1e-18:
        return [float(x_min)]

    gap = max(AXIS_LABEL_GAP_PX, width_px / 11.0)

    def vx(v: float) -> float:
        return (v - x_min) / (span + 1e-18) * width_px

    label_steps = (0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)

    for lab_step in label_steps:
        grid_vals = _enumerate_ticks(lab_step, x_min, x_max)
        if len(grid_vals) > 36:
            continue

        cand: list[tuple[float, float, int]] = []
        for v in grid_vals:
            cand.append((float(v), 40))
        cand.append((float(x_min), 90))
        cand.append((float(x_max), 90))
        if x_min <= 0.0 <= x_max:
            cand.append((0.0, 100))

        by_val: dict[float, tuple[float, int]] = {}
        for val, pr in cand:
            key = round(val, 10)
            if key not in by_val or pr > by_val[key][1]:
                by_val[key] = (val, pr)

        entries = [(vx(v), v, pr) for v, pr in by_val.values()]
        entries.sort(key=lambda z: (-z[2], z[0]))

        picked: list[tuple[float, float, int]] = []
        for px0, val, pr in entries:
            px = float(np.clip(px0, 0.0, width_px))
            if all(abs(px - p1) >= gap - 0.5 for p1, _, _ in picked):
                picked.append((px, val, pr))

        if len(picked) > AXIS_MAX_LABELS:
            picked.sort(key=lambda z: z[2])
            while len(picked) > AXIS_MAX_LABELS:
                picked.pop(0)

        return sorted({round(t[1], 12) for t in picked})

    return [float(x_min), float(x_max)]


def _fmt_uplift_tick_label(ratio: float) -> str:
    p = ratio * 100.0
    if abs(p) < 1e-9:
        return "0%"
    if p > 0:
        return f"+{p:.0f}%"
    return f"{p:.0f}%"


def build_global_uplift_axis_html(
    x_min: float,
    x_max: float,
    *,
    width_px: int = 320,
    height_px: int = 38,
) -> str:
    """
    Shared uplift axis above all distribution strips (same vx as charts).
    GrowthBook-like: %-grid ticks, subtle stems, strong 0% reference, muted labels.
    """
    w_f, h_f = float(width_px), float(height_px)
    gid = abs(hash((round(x_min, 9), round(x_max, 9), width_px, height_px))) % 800000 + 100000

    spine_y = 12.0
    minor_tick_top = 3.0
    zero_tick_top = 1.5
    label_y = 27.0
    fade_w = min(44.0, w_f * 0.14)

    def vx(val: float) -> float:
        return (val - x_min) / (x_max - x_min + 1e-15) * w_f

    minor_ticks = _axis_minor_ticks(x_min, x_max)
    label_vals = _axis_sparse_labels(x_min, x_max, w_f)
    zero_px = float(np.clip(vx(0.0), 0.0, w_f))

    minor_lines: list[str] = []
    labels: list[str] = []

    for tv in minor_ticks:
        if abs(tv) < 1e-12:
            continue
        tx = float(np.clip(vx(tv), 0.0, w_f))
        minor_lines.append(
            f'<line x1="{tx:.3f}" y1="{spine_y:.2f}" x2="{tx:.3f}" y2="{minor_tick_top:.2f}" '
            f'stroke="{GB_AXIS_TICK_SUBTLE}" stroke-width="1" stroke-linecap="square" />'
        )

    for lv in label_vals:
        tx = float(np.clip(vx(lv), 0.0, w_f))
        is_zero = abs(lv) < 1e-12
        lbl = _fmt_uplift_tick_label(lv)
        fill = GB_AXIS_LABEL_STRONG if is_zero else GB_AXIS_LABEL
        weight = "600" if is_zero else "400"
        labels.append(
            f'<text x="{tx:.3f}" y="{label_y:.1f}" text-anchor="middle" dominant-baseline="middle" '
            f'fill="{fill}" font-size="{AXIS_LABEL_FONT_PX}" font-weight="{weight}" '
            f'font-family="Inter, -apple-system, BlinkMacSystemFont, sans-serif">'
            f"{html.escape(lbl)}</text>"
        )

    zero_line = ""
    if x_min <= 0.0 <= x_max:
        zero_line = (
            f'<line x1="{zero_px:.3f}" y1="{spine_y:.2f}" x2="{zero_px:.3f}" y2="{zero_tick_top:.2f}" '
            f'stroke="{GB_ZERO_LINE}" stroke-width="2.75" stroke-linecap="square" '
            "style=\"filter: drop-shadow(0 0 4px rgba(59,130,246,0.35));\" />"
        )

    label_top = spine_y + 2.0
    label_h = max(h_f - label_top, 14.0)

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {w_f:.0f} {h_f:.0f}" preserveAspectRatio="none">
  <defs>
    <linearGradient id="gbAxFadeL{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{fade_w:.1f}" y2="0">
      <stop offset="0%" stop-color="{GB_PANEL_BG}" stop-opacity="0.92"/>
      <stop offset="100%" stop-color="{GB_PANEL_BG}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="gbAxFadeR{gid}" gradientUnits="userSpaceOnUse" x1="{w_f:.1f}" y1="0" x2="{w_f - fade_w:.1f}" y2="0">
      <stop offset="0%" stop-color="{GB_PANEL_BG}" stop-opacity="0.92"/>
      <stop offset="100%" stop-color="{GB_PANEL_BG}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <line x1="0" y1="{spine_y:.2f}" x2="{w_f:.0f}" y2="{spine_y:.2f}" stroke="{GB_CELL_LINE}" stroke-width="1" opacity="0.65" />
  {"".join(minor_lines)}
  {zero_line}
  {"".join(labels)}
  <rect x="0" y="{label_top:.2f}" width="{fade_w:.1f}" height="{label_h:.2f}" fill="url(#gbAxFadeL{gid})" style="pointer-events:none"/>
  <rect x="{w_f - fade_w:.1f}" y="{label_top:.2f}" width="{fade_w:.1f}" height="{label_h:.2f}" fill="url(#gbAxFadeR{gid})" style="pointer-events:none"/>
</svg>
"""
    return f'<div class="gb-global-dist-axis">{svg}</div>'


def _tooltip_pct_signed(ratio: float) -> str:
    """One signed percent for CI / uplift lines; em dash when undefined."""
    if not np.isfinite(ratio):
        return "—"
    return f"{ratio * 100:+.1f}%"


def _tooltip_uplift_color(uplift: float) -> str:
    if not np.isfinite(uplift):
        return "#9CA3AF"
    if uplift > 0:
        return "#22c55e"
    if uplift < 0:
        return "#ef4444"
    return "#9CA3AF"


def _tooltip_probability_color(probability: float) -> str:
    if not np.isfinite(probability):
        return "#9CA3AF"
    if probability > 0.95:
        return "#22c55e"
    if probability < 0.6:
        return "#ef4444"
    return "#9CA3AF"


def _build_distribution_tooltip_html(
    metric: str,
    uplift: float,
    lo_u: float,
    hi_u: float,
    probability: float,
) -> str:
    """Uplift is deterministic (same as table); lo_u/hi_u from bootstrap distribution."""
    metric_esc = html.escape(metric)
    upl_c = _tooltip_uplift_color(uplift)
    pr_c = _tooltip_probability_color(probability)
    if np.isfinite(uplift):
        uplift_inner = f'<span style="color:{upl_c};">{uplift * 100:+.1f}%</span>'
    else:
        uplift_inner = '<span style="color:#9CA3AF;">—</span>'
    ci_segment = (
        f'{_tooltip_pct_signed(lo_u)}, {_tooltip_pct_signed(hi_u)}'
    )
    ci_esc = html.escape(ci_segment)
    if np.isfinite(probability):
        prob_inner = (
            f'<span style="color:{pr_c};">{probability * 100:.1f}%</span>'
        )
    else:
        prob_inner = '<span style="color:#9CA3AF;">—</span>'
    return f"""<div>
  <div style="font-weight:600;">{metric_esc}</div>
  <div>
    Uplift:
    {uplift_inner}
  </div>
  <div style="color:#9CA3AF;">
    95% CI: [{ci_esc}]
  </div>
  <div>
    Chance to win:
    {prob_inner}
  </div>
</div>"""


def _dist_svg_wrap(inner_svg_markup: str, w_f: float, h_f: float) -> str:
    """Outer SVG + group; identical drawing to prior version without title/hit-layer."""
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {w_f:.0f} {h_f:.0f}" preserveAspectRatio="none">
  <g>
{inner_svg_markup}
  </g>
</svg>
"""


def _distribution_bar_shell(svg_markup: str, tooltip_inner_html: str) -> str:
    """Hover target covers full bar; tooltip outside SVG."""
    return f"""<div class="dist-wrap">
  <div class="dist-bar">
    <div class="density-layer">{svg_markup}</div>
  </div>
  <div class="gb-tooltip tooltip">{tooltip_inner_html}</div>
</div>"""


def build_distribution_bar_html(
    uplift_samples: np.ndarray,
    *,
    metric_name: str,
    uplift: float,
    probability: float,
    x_min: float,
    x_max: float,
    width_px: int = 320,
    height_px: int = 40,
    ci: float = 0.95,
) -> str:
    """
    Relative uplift density on a fixed global x-range (same vx mapping as top axis).
    Bootstrap draws define the KDE + 95% CI only.
    Dashed line and tooltip uplift use deterministic (mean_B - mean_A) / mean_A — same as the table.
    """
    w_f, h_f = float(width_px), float(height_px)
    pad_b = 4.0
    pad_t = 2.5
    avail_h = h_f - pad_b - pad_t
    baseline_y = h_f - pad_b

    u = np.asarray(uplift_samples, dtype=float)
    u = u[np.isfinite(u)]

    upl_det = float(uplift)

    if u.size >= 8:
        lo_u, hi_u = np.percentile(
            u, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
        )
        lo_u, hi_u = float(lo_u), float(hi_u)
    else:
        lo_u = float("nan")
        hi_u = float("nan")

    tooltip_html = _build_distribution_tooltip_html(
        metric_name, upl_det, lo_u, hi_u, float(probability)
    )

    def vx(val: float) -> float:
        return (val - x_min) / (x_max - x_min + 1e-15) * w_f

    zero_px = float(np.clip(vx(0.0), 0.0, w_f))

    empty_markup = (
        f'    <line x1="{zero_px:.3f}" y1="0" x2="{zero_px:.3f}" y2="{h_f:.0f}" '
        f'stroke="{GB_ZERO_LINE}" stroke-width="1.75" />'
    )
    empty_svg = _dist_svg_wrap(empty_markup, w_f, h_f)

    empty_bar = _distribution_bar_shell(empty_svg, tooltip_html)

    if u.size < 8:
        return empty_bar

    n_pts = 200
    x_lin = np.linspace(x_min, x_max, n_pts)

    dens = _eval_smoothed_density(u, x_lin)
    dens = _squash_peak(dens, gamma=0.56)
    y_top = baseline_y - dens * avail_h * 0.94
    y_top = np.clip(y_top, pad_t, baseline_y)

    x_px = np.array([vx(float(x)) for x in x_lin], dtype=float)
    ci_lo_px = float(min(vx(float(lo_u)), vx(float(hi_u))))
    ci_hi_px = float(max(vx(float(lo_u)), vx(float(hi_u))))
    ci_w = max(ci_hi_px - ci_lo_px, 1.0)

    path_left, path_right = _split_uplift_fill_paths(x_lin, y_top, baseline_y, x_px, zero_px)

    path_layers: list[str] = []
    if path_left:
        path_layers.append(
            f'<path d="{path_left}" fill="rgba(239,68,68,0.5)" stroke="rgba(239,68,68,0.35)" '
            f'stroke-width="0.45" stroke-linejoin="round" />'
        )
    if path_right:
        path_layers.append(
            f'<path d="{path_right}" fill="rgba(34,197,94,0.5)" stroke="rgba(34,197,94,0.35)" '
            f'stroke-width="0.45" stroke-linejoin="round" />'
        )

    mean_line_svg = ""
    if np.isfinite(upl_det):
        mpx = float(np.clip(vx(upl_det), 0.0, w_f))
        mean_line_svg = (
            f'    <line x1="{mpx:.3f}" y1="2" x2="{mpx:.3f}" y2="{baseline_y:.3f}"\n'
            f'        stroke="rgba(255,255,255,0.9)" stroke-width="1.35" stroke-dasharray="4 3" />'
        )

    chart_markup = (
        f'    <rect x="{ci_lo_px:.3f}" y="0" width="{ci_w:.3f}" height="{h_f:.0f}" '
        f'fill="rgba(148,163,184,0.16)" />\n'
        f'  {"".join(path_layers)}\n'
        f'    <line x1="{zero_px:.3f}" y1="0" x2="{zero_px:.3f}" y2="{baseline_y:.3f}" '
        f'stroke="{GB_ZERO_LINE}" stroke-width="1.75" />\n'
        f"{mean_line_svg}"
    )
    chart_svg = _dist_svg_wrap(chart_markup, w_f, h_f)

    return _distribution_bar_shell(chart_svg, tooltip_html)


def render_goal_metrics_shell_open_html() -> str:
    return '<div class="gb-metrics-shell">'


def render_goal_metrics_column_headers_html() -> str:
    """Column title row (must sit inside `.gb-metrics-shell`)."""
    return """
  <div class="gb-row gb-header-row">
    <div class="gb-cell gb-metric gb-metric--sticky-col gb-hdr">Metric</div>
    <div class="gb-cell gb-base gb-hdr">Baseline</div>
    <div class="gb-cell gb-var gb-hdr">Variation</div>
    <div class="gb-cell gb-chance gb-hdr">Chance to win</div>
    <div class="gb-cell gb-dist gb-hdr">Distribution</div>
    <div class="gb-cell gb-pct gb-hdr">% Change</div>
  </div>
"""


def metric_role_tag_class(role: str) -> str:
    """CSS class for metric role (matches `.primary-metric` + `tag--*` in app + GrowthBook)."""
    return {
        "Primary": "primary-metric",
        "Secondary": "tag--secondary",
        "Guardrail": "tag--guardrail",
        "Other": "tag--other",
    }.get(role, "tag--other")


def render_goal_metrics_global_axis_row_html(x_min: float, x_max: float) -> str:
    """Shared uplift axis for the Distribution column (same xlim as all strip charts)."""
    axis = build_global_uplift_axis_html(x_min, x_max)
    return f"""
<div class="gb-row gb-axis-row">
  <div class="gb-cell gb-metric gb-metric--sticky-col"></div>
  <div class="gb-cell gb-base"></div>
  <div class="gb-cell gb-var"></div>
  <div class="gb-cell gb-chance"></div>
  <div class="gb-cell gb-dist">{axis}</div>
  <div class="gb-cell gb-pct"></div>
</div>
"""


def render_metric_row(
    ma: MetricAnalysis,
    *,
    uplift_samples: np.ndarray,
    dist_x_min: float,
    dist_x_max: float,
    c_arm: np.ndarray,
    t_arm: np.ndarray,
    alpha: float,
) -> str:
    """
    Returns one HTML row (flex) for the GrowthBook metrics table.
    Precompute uplift_samples via bootstrap_uplift_samples(...).
    """
    base_main, base_sub = format_arm_cell(ma, c_arm, is_control=True)
    var_main, var_sub = format_arm_cell(ma, t_arm, is_control=False)

    pct_html = format_pct_change_html(ma.uplift)
    chance_color = chance_to_win_style(ma.probability)

    sig = ma.p_value < alpha and not np.isnan(ma.uplift)
    if sig and ma.uplift > 0:
        sig_badge = (
            f'<span style="font-size:11px;color:{GB_POSITIVE};font-weight:600;"> ●</span>'
        )
    elif sig and ma.uplift < 0:
        sig_badge = (
            f'<span style="font-size:11px;color:{GB_NEGATIVE};font-weight:600;"> ●</span>'
        )
    else:
        sig_badge = ""

    bar_html = build_distribution_bar_html(
        uplift_samples,
        metric_name=ma.metric,
        uplift=float(ma.uplift),
        probability=float(ma.probability),
        x_min=dist_x_min,
        x_max=dist_x_max,
    )
    name_esc = html.escape(ma.metric)
    role_esc = html.escape(ma.role)
    kind_esc = html.escape(ma.kind)
    role_tag = metric_role_tag_class(ma.role)

    return f"""
<div class="gb-row">
  <div class="gb-cell gb-metric gb-metric--sticky-col">
    <div class="gb-metric-title {role_tag}">{name_esc}{sig_badge}</div>
    <div style="color:{GB_TEXT_SECONDARY};font-size:12px;margin-top:4px;line-height:1.4;font-family:{APP_FONT_STACK};">{role_esc} · {kind_esc}</div>
  </div>
  <div class="gb-cell gb-base">
    <div style="color:{GB_TEXT_PRIMARY};font-weight:500;font-size:15px;line-height:1.4;font-family:{APP_FONT_STACK};">{html.escape(base_main)}</div>
    <div style="color:{GB_TEXT_SECONDARY};font-size:12px;margin-top:4px;line-height:1.4;font-family:{APP_FONT_STACK};">{html.escape(base_sub)}</div>
  </div>
  <div class="gb-cell gb-var">
    <div style="color:{GB_TEXT_PRIMARY};font-weight:500;font-size:15px;line-height:1.4;font-family:{APP_FONT_STACK};">{html.escape(var_main)}</div>
    <div style="color:{GB_TEXT_SECONDARY};font-size:12px;margin-top:4px;line-height:1.4;font-family:{APP_FONT_STACK};">{html.escape(var_sub)}</div>
  </div>
  <div class="gb-cell gb-chance">
    <span style="color:{chance_color};font-weight:600;font-size:18px;line-height:1.4;font-family:{APP_FONT_STACK};">{ma.probability * 100:.1f}%</span>
  </div>
  <div class="gb-cell gb-dist">{bar_html}</div>
  <div class="gb-cell gb-pct"><div style="padding-top:4px;font-size:15px;line-height:1.4;font-family:{APP_FONT_STACK};">{pct_html}</div></div>
</div>
"""


def close_goal_metrics_shell_html() -> str:
    return "</div>"


def build_goal_metrics_panel_html(
    analyses: list[MetricAnalysis],
    df: pd.DataFrame,
    control: str,
    treatment: str,
    alpha: float,
) -> str:
    """Full HTML/CSS/SVG panel (iframe-safe for Streamlit)."""
    role_order = ("Primary", "Secondary", "Guardrail", "Other")
    section_label = {
        "Primary": "PRIMARY METRIC",
        "Secondary": "SECONDARY METRICS",
        "Guardrail": "GUARDRAIL METRICS",
        "Other": "OTHER METRICS",
    }
    section_wrap_class = {
        "Primary": "gb-section-wrap gb-section-primary",
        "Secondary": "gb-section-wrap gb-section-secondary",
        "Guardrail": "gb-section-wrap gb-section-guardrail",
        "Other": "gb-section-wrap gb-section-other",
    }

    rows_by_role: dict[str, list[tuple[MetricAnalysis, np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    for ma in analyses:
        arms = extract_metric_arms(df, ma.metric, control, treatment)
        if arms is None:
            continue
        c_arm, t_arm, binary = arms
        uplift_samples = bootstrap_uplift_samples(c_arm, t_arm, binary=binary)
        rows_by_role[ma.role].append((ma, c_arm, t_arm, uplift_samples))

    def _abs_uplift_sort_key(
        row: tuple[MetricAnalysis, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple:
        u = row[0].uplift
        if np.isnan(u):
            return (1, 0.0, row[0].metric)
        return (0, -abs(float(u)), row[0].metric)

    for role in role_order:
        rows_by_role[role].sort(key=_abs_uplift_sort_key)

    uplift_arrays: list[np.ndarray] = []
    for role in role_order:
        for _, _, _, u in rows_by_role.get(role, []):
            uplift_arrays.append(u)
    gx_min, gx_max = compute_global_uplift_xlim(uplift_arrays)

    parts: list[str] = [
        GROWTHBOOK_METRICS_STYLES,
        render_goal_metrics_shell_open_html(),
        '<div class="gb-table-wrapper">',
        '<div class="gb-table-header">',
        render_goal_metrics_column_headers_html(),
        render_goal_metrics_global_axis_row_html(gx_min, gx_max),
        "</div>",
        '<div class="gb-table-body">',
    ]
    first_section = True
    for role in role_order:
        block = rows_by_role.get(role, [])
        if not block:
            continue
        head_cls = "gb-section-head gb-section-head-first" if first_section else "gb-section-head"
        first_section = False
        parts.append(f'<div class="{html.escape(section_wrap_class[role])}">')
        parts.append(
            f'<div class="{head_cls}">{html.escape(section_label[role])}</div>'
            '<div class="gb-section-divider"></div>'
        )
        for ma, c_arm, t_arm, uplift_samples in block:
            parts.append(
                render_metric_row(
                    ma,
                    uplift_samples=uplift_samples,
                    dist_x_min=gx_min,
                    dist_x_max=gx_max,
                    c_arm=c_arm,
                    t_arm=t_arm,
                    alpha=alpha,
                )
            )
        parts.append("</div>")
    parts.append("</div></div>")
    parts.append(close_goal_metrics_shell_html())
    return "\n".join(parts)


def analyze_two_arm_metric(
    df: pd.DataFrame,
    metric: str,
    role: str,
    control_label: str,
    treatment_label: str,
) -> MetricAnalysis | None:
    work = df[["variant", metric]].copy()
    work["variant"] = work["variant"].map(_norm_variant)
    c = work[work["variant"] == control_label][metric].dropna()
    t = work[work["variant"] == treatment_label][metric].dropna()

    if c.empty or t.empty:
        return None

    n_c, n_t = len(c), len(t)
    binary = is_binary_metric(work[metric])

    if binary:
        conv_c = int(c.astype(float).sum())
        conv_t = int(t.astype(float).sum())
        rate_c = conv_c / n_c
        rate_t = conv_t / n_t
        count = np.array([conv_c, conv_t])
        nobs = np.array([n_c, n_t])
        stat, p_value = proportions_ztest(count, nobs)
        prob = prob_b_better_binary(conv_c, n_c, conv_t, n_t)
        return MetricAnalysis(
            metric=metric,
            role=role,
            kind="binary",
            n_control=n_c,
            n_treatment=n_t,
            value_control=rate_c,
            value_treatment=rate_t,
            uplift=relative_uplift(rate_c, rate_t),
            p_value=float(p_value),
            probability=prob,
        )

    v_c = float(c.astype(float).mean())
    v_t = float(t.astype(float).mean())
    t_stat, p_value = ttest_ind(
        c.astype(float), t.astype(float), equal_var=False, nan_policy="omit"
    )
    prob = prob_b_better_continuous(c.astype(float).values, t.astype(float).values)
    return MetricAnalysis(
        metric=metric,
        role=role,
        kind="continuous",
        n_control=n_c,
        n_treatment=n_t,
        value_control=v_c,
        value_treatment=v_t,
        uplift=relative_uplift(v_c, v_t),
        p_value=float(p_value),
        probability=prob,
    )


def assign_roles(
    all_metrics: list[str],
    primary: str | None,
    secondary: list[str],
    guardrails: list[str],
) -> dict[str, str]:
    """Map metric -> role string. Unlisted metrics -> Other."""
    roles: dict[str, str] = {}
    used: set[str] = set()
    if primary:
        roles[primary] = "Primary"
        used.add(primary)
    for m in secondary:
        if m not in used:
            roles[m] = "Secondary"
            used.add(m)
    for m in guardrails:
        if m not in used:
            roles[m] = "Guardrail"
            used.add(m)
    for m in all_metrics:
        if m not in used:
            roles[m] = "Other"
            used.add(m)
    return roles


# --- CSV tab: metrics roles (session state, mutually exclusive) -----------------

CSV_METRICS_CONFIG_KEY = "csv_metrics_config"
CSV_METRICS_FP_KEY = "csv_metrics_fingerprint"
CSV_ALL_METRICS_KEY = "csv_all_metrics_list"
PRIMARY_WIDGET_NONE = "— No primary metric —"


def _csv_widget_primary_to_metric(v: str) -> str | None:
    return None if v == PRIMARY_WIDGET_NONE else v


def _csv_metric_to_widget_primary(p: str | None) -> str:
    return PRIMARY_WIDGET_NONE if p is None else p


def set_primary_exclusive(
    cfg: dict[str, Any], new_primary: str | None, all_metrics: list[str]
) -> None:
    valid = set(all_metrics)
    if new_primary is not None and new_primary not in valid:
        return
    cfg["primary"] = new_primary
    if new_primary is not None:
        cfg["secondary"] = [m for m in cfg["secondary"] if m != new_primary]
        cfg["guardrail"] = [m for m in cfg["guardrail"] if m != new_primary]


def _normalize_cfg_triple_inplace(cfg: dict[str, Any], all_metrics: list[str]) -> None:
    """
    Enforce exclusivity: unknown names dropped; secondary vs guardrail → secondary wins;
    if primary matches a secondary or guardrail choice → unset primary; then primary
    removes that metric from the other lists.
    """
    valid = set(all_metrics)
    p = cfg.get("primary")
    if p is not None and p not in valid:
        p = None
    s = [m for m in cfg.get("secondary", []) if m in valid]
    g = [m for m in cfg.get("guardrail", []) if m in valid]
    s_set = set(s)
    g_set = set(g)
    if p is not None and p in s_set:
        p = None
    if p is not None and p in g_set:
        p = None
    if p is not None:
        s = [m for m in s if m != p]
        g = [m for m in g if m != p]
    s_set = set(s)
    g = [m for m in g if m not in s_set]
    g_set = set(g)
    s = [m for m in s if m not in g_set]
    cfg["primary"] = p
    cfg["secondary"] = s
    cfg["guardrail"] = g


def migrate_csv_metrics_config(cfg: dict[str, Any], all_metrics: list[str]) -> None:
    """Keep assignments that still exist; drop stale names; re-enforce exclusivity."""
    if not all_metrics:
        cfg["primary"] = None
        cfg["secondary"] = []
        cfg["guardrail"] = []
        return
    valid = set(all_metrics)
    cfg["secondary"] = [m for m in cfg.get("secondary", []) if m in valid]
    cfg["guardrail"] = [m for m in cfg.get("guardrail", []) if m in valid]
    p = cfg.get("primary")
    cfg["primary"] = p if p in valid else None
    _normalize_cfg_triple_inplace(cfg, all_metrics)
    if cfg["primary"] is None:
        taken = set(cfg["secondary"]) | set(cfg["guardrail"])
        for m in all_metrics:
            if m not in taken:
                cfg["primary"] = m
                break
        if cfg["primary"] is None:
            cfg["primary"] = all_metrics[0]
        set_primary_exclusive(cfg, cfg["primary"], all_metrics)


def _sync_csv_metrics_widget_keys() -> None:
    cfg = st.session_state[CSV_METRICS_CONFIG_KEY]
    st.session_state["csv_w_primary"] = _csv_metric_to_widget_primary(cfg["primary"])
    st.session_state["csv_w_secondary"] = list(cfg["secondary"])
    st.session_state["csv_w_guardrail"] = list(cfg["guardrail"])


def init_csv_metrics_state(all_metrics: list[str]) -> dict[str, Any]:
    """Single source of truth: metrics_config in session_state; widgets mirror it."""
    st.session_state[CSV_ALL_METRICS_KEY] = list(all_metrics)
    fp = tuple(all_metrics)
    if CSV_METRICS_CONFIG_KEY not in st.session_state:
        st.session_state[CSV_METRICS_CONFIG_KEY] = {
            "primary": all_metrics[0] if all_metrics else None,
            "secondary": [],
            "guardrail": [],
        }
    if st.session_state.get(CSV_METRICS_FP_KEY) != fp:
        st.session_state[CSV_METRICS_FP_KEY] = fp
        migrate_csv_metrics_config(st.session_state[CSV_METRICS_CONFIG_KEY], all_metrics)
        _sync_csv_metrics_widget_keys()
    elif "csv_w_primary" not in st.session_state:
        _sync_csv_metrics_widget_keys()
    return st.session_state[CSV_METRICS_CONFIG_KEY]


def _csv_metrics_pre_widget_reconcile(cfg: dict[str, Any], all_metrics: list[str]) -> None:
    """
    Merge widget session values into metrics_config before widgets render (so we can
    fix keys in the same run without Streamlit's post-widget mutation error).
    """
    if "csv_w_primary" not in st.session_state:
        return
    tmp = {
        "primary": _csv_widget_primary_to_metric(st.session_state["csv_w_primary"]),
        "secondary": [
            m
            for m in st.session_state.get("csv_w_secondary", [])
            if m in all_metrics
        ],
        "guardrail": [
            m
            for m in st.session_state.get("csv_w_guardrail", [])
            if m in all_metrics
        ],
    }
    _normalize_cfg_triple_inplace(tmp, all_metrics)
    if (
        cfg["primary"] != tmp["primary"]
        or cfg["secondary"] != tmp["secondary"]
        or cfg["guardrail"] != tmp["guardrail"]
    ):
        cfg["primary"] = tmp["primary"]
        cfg["secondary"] = tmp["secondary"]
        cfg["guardrail"] = tmp["guardrail"]
        st.session_state["csv_w_primary"] = _csv_metric_to_widget_primary(cfg["primary"])
        st.session_state["csv_w_secondary"] = list(cfg["secondary"])
        st.session_state["csv_w_guardrail"] = list(cfg["guardrail"])


def csv_secondary_multiselect_options(
    cfg: dict[str, Any], all_metrics: list[str]
) -> list[str]:
    p = cfg["primary"]
    blocked = ({p} if p else set()) | set(cfg["guardrail"])
    return sorted((set(all_metrics) - blocked) | set(cfg["secondary"]))


def csv_guardrail_multiselect_options(
    cfg: dict[str, Any], all_metrics: list[str]
) -> list[str]:
    p = cfg["primary"]
    blocked = ({p} if p else set()) | set(cfg["secondary"])
    return sorted((set(all_metrics) - blocked) | set(cfg["guardrail"]))


def detect_conflicts(
    primary: MetricAnalysis | None,
    analyses_by_metric: dict[str, MetricAnalysis],
    secondary_metric_names: list[str],
) -> list[str]:
    msgs: list[str] = []
    if primary is None:
        return msgs
    pu = primary.uplift
    if np.isnan(pu):
        return msgs
    for name in secondary_metric_names:
        other = analyses_by_metric.get(name)
        if other is None or np.isnan(other.uplift):
            continue
        ou = other.uplift
        if pu > 0 and ou < 0:
            msgs.append(
                f"Primary metric **{primary.metric}** is up ({pu:+.1%} vs control) but "
                f"secondary **{name}** is down ({ou:+.1%})."
            )
        elif pu < 0 and ou > 0:
            msgs.append(
                f"Primary metric **{primary.metric}** is down ({pu:+.1%}) but "
                f"secondary **{name}** is up ({ou:+.1%})."
            )
    rev = analyses_by_metric.get("revenue")
    if rev is not None and primary.metric != "revenue":
        if not np.isnan(pu) and not np.isnan(rev.uplift):
            if pu > 0 > rev.uplift:
                msgs.append(
                    f"Primary **{primary.metric}** improves ({pu:+.1%}) while **revenue** "
                    f"declines ({rev.uplift:+.1%})."
                )
    return list(dict.fromkeys(msgs))


def compute_decision(
    alpha: float,
    primary: MetricAnalysis | None,
    guardrail_analyses: list[MetricAnalysis],
    conflict_messages: list[str],
) -> tuple[str, list[str]]:
    """
    Returns status label and bullet reasons.
    Precedence: WARNING (guardrail harm) -> UNCERTAIN (conflicts) -> INCONCLUSIVE ->
    POSITIVE / NEGATIVE.
    """
    reasons: list[str] = []

    worsened_guardrails = [
        g
        for g in guardrail_analyses
        if g.p_value < alpha and g.uplift < 0 and not np.isnan(g.uplift)
    ]
    if worsened_guardrails:
        for g in worsened_guardrails:
            reasons.append(
                f"Guardrail **{g.metric}** worsened significantly "
                f"(uplift {g.uplift:+.1%}, p={g.p_value:.4f})."
            )
        return "WARNING", reasons

    if conflict_messages:
        reasons.extend(conflict_messages)
        return "UNCERTAIN", reasons

    if primary is None:
        return "INCONCLUSIVE", ["No primary metric configured or analyzable."]

    if np.isnan(primary.uplift):
        return "INCONCLUSIVE", ["Primary metric uplift could not be computed (e.g. zero baseline)."]

    primary_sig = primary.p_value < alpha
    if not primary_sig:
        reasons.append(
            f"Primary **{primary.metric}** not significant at α={alpha} "
            f"(p={primary.p_value:.4f})."
        )
        return "INCONCLUSIVE", reasons

    if primary.uplift > 0:
        reasons.append(
            f"Primary **{primary.metric}** significant uplift {primary.uplift:+.1%} "
            f"(p={primary.p_value:.4f})."
        )
        return "POSITIVE", reasons

    reasons.append(
        f"Primary **{primary.metric}** significant but uplift {primary.uplift:+.1%} "
        f"(p={primary.p_value:.4f})."
    )
    return "NEGATIVE", reasons


def sort_analyses_by_role(analyses: list[MetricAnalysis]) -> list[MetricAnalysis]:
    """Primary → Secondary → Guardrail → Other; within each role sort by |uplift| DESC."""
    role_rank = {"Primary": 0, "Secondary": 1, "Guardrail": 2, "Other": 3}

    def _key(a: MetricAnalysis) -> tuple:
        u = a.uplift
        if np.isnan(u):
            nan_grp = 1
            abs_key = 0.0
        else:
            nan_grp = 0
            abs_key = -abs(float(u))
        return (role_rank.get(a.role, 9), nan_grp, abs_key, a.metric)

    return sorted(analyses, key=_key)


def segment_filter_widget_key(prop: str) -> str:
    return "exp_sf_" + hashlib.sha256(str(prop).encode()).hexdigest()[:16]


def candidate_user_property_columns(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Columns that may be used as segment dimensions (not id / variant / date / metrics)."""
    metrics_set = set(metrics)
    out: list[str] = []
    for c in df.columns:
        if c in metrics_set:
            continue
        if str(c).strip().lower() in SEGMENT_EXCLUDED_COLUMN_NAMES:
            continue
        out.append(c)
    return sorted(out, key=str)


def property_uniques_for_segment(base_df: pd.DataFrame, prop: str) -> list[Any]:
    if prop not in base_df.columns:
        return []
    ser = base_df[prop].dropna()
    vals = ser.unique().tolist()
    vals.sort(key=lambda x: str(x))
    return vals


def run_metric_analyses(
    work: pd.DataFrame,
    metrics: list[str],
    roles_map: dict[str, str],
    control: str,
    treatment: str,
) -> list[MetricAnalysis]:
    analyses: list[MetricAnalysis] = []
    for m in metrics:
        ma = analyze_two_arm_metric(work, m, roles_map[m], control, treatment)
        if ma:
            analyses.append(ma)
    return analyses


def exp_mark_segment_reapply() -> None:
    st.session_state[EXP_SEGMENT_REAPPLY_KEY] = True


# --- Streamlit app ------------------------------------------------------------

st.set_page_config(page_title="Experiment Analyzer", layout="wide")
st.markdown(GLOBAL_APP_STYLES, unsafe_allow_html=True)
st.title("📊 Experiment Decision Support System")

st.markdown("### 📂 Upload experiment data")

with st.container():
    st.markdown(
        '<span class="csv-upload-marker" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        label="\u200b",
        type=["csv"],
        help="File should have one row per user",
        label_visibility="collapsed",
        key="csv_multi",
    )

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

alpha = st.sidebar.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.caption(f"Rows: **{len(df_raw)}** · Columns: **{', '.join(map(str, df_raw.columns))}**")
    st.dataframe(df_raw.head(20), use_container_width=True)

    if "variant" not in df_raw.columns:
        st.error("CSV must include a **variant** column.")
    else:
        df = df_raw.copy()
        df["variant"] = df["variant"].map(_norm_variant)
        metrics = detect_metric_columns(df)
        upload_name = getattr(uploaded_file, "name", "") or str(id(uploaded_file))
        data_fp = (tuple(str(c) for c in df.columns), upload_name, len(df))
        if st.session_state.get(EXP_DATA_FP_KEY) != data_fp:
            st.session_state[EXP_DATA_FP_KEY] = data_fp
            st.session_state.pop(EXP_RESULTS_READY_KEY, None)
            st.session_state.pop(EXP_BASE_DF_KEY, None)
            st.session_state.pop(EXP_WORK_DF_KEY, None)
            st.session_state.pop(EXP_ANALYSES_KEY, None)
            st.session_state.pop(EXP_METRICS_SNAP_KEY, None)
            st.session_state.pop(EXP_CONTROL_SNAP_KEY, None)
            st.session_state.pop(EXP_TREATMENT_SNAP_KEY, None)
            st.session_state.pop(EXP_SEGMENT_REAPPLY_KEY, None)
            for _rk in list(st.session_state.keys()):
                if isinstance(_rk, str) and _rk.startswith("exp_sf_"):
                    st.session_state.pop(_rk, None)
            if "exp_selected_properties" in st.session_state:
                del st.session_state["exp_selected_properties"]
        if metrics:
            st.info("**Detected metrics (numeric columns):** " + ", ".join(metrics))

        uniq_variants = sorted(df["variant"].unique())
        c1, c2 = st.columns(2)
        ctrl_idx = uniq_variants.index(DEFAULT_CONTROL) if DEFAULT_CONTROL in uniq_variants else 0
        control = c1.selectbox("Control variant", uniq_variants, index=ctrl_idx)
        treatment_choices = [v for v in uniq_variants if v != control]
        if not treatment_choices:
            st.error("Need at least two distinct variants to compare.")
        else:
            default_trt = (
                DEFAULT_TREATMENT
                if DEFAULT_TREATMENT in treatment_choices
                else treatment_choices[0]
            )
            trt_idx = treatment_choices.index(default_trt)
            treatment = c2.selectbox(
                "Treatment variant",
                treatment_choices,
                index=trt_idx,
            )

            st.header("2. Configure metrics")
            if not metrics:
                st.warning(
                    "No numeric metric columns detected (after excluding ids / dates)."
                )
            else:
                metrics_config = init_csv_metrics_state(metrics)
                _csv_metrics_pre_widget_reconcile(metrics_config, metrics)
                primary_choices = [PRIMARY_WIDGET_NONE] + sorted(metrics)
                # Chip/tint colors are scoped in CSS from these per-block markers (not from metric data).
                with st.container():
                    st.markdown(
                        '<div class="role-marker role-marker--primary"></div>',
                        unsafe_allow_html=True,
                    )
                    st.selectbox(
                        "Primary metric (one)",
                        options=primary_choices,
                        key="csv_w_primary",
                        help="Choosing a metric here moves it out of secondary / guardrail.",
                    )
                with st.container():
                    st.markdown(
                        '<div class="role-marker role-marker--secondary"></div>',
                        unsafe_allow_html=True,
                    )
                    st.multiselect(
                        "Secondary metrics",
                        options=csv_secondary_multiselect_options(
                            metrics_config, metrics
                        ),
                        key="csv_w_secondary",
                        help="Metrics already used as primary or guardrail are hidden; "
                        "adding one here removes it from guardrail.",
                    )
                with st.container():
                    st.markdown(
                        '<div class="role-marker role-marker--guardrail"></div>',
                        unsafe_allow_html=True,
                    )
                    st.multiselect(
                        "Guardrail metrics",
                        options=csv_guardrail_multiselect_options(
                            metrics_config, metrics
                        ),
                        key="csv_w_guardrail",
                        help="Metrics already used as primary or secondary are hidden; "
                        "adding one here removes it from secondary.",
                    )
                st.caption(
                    "Any metric not chosen above is classified as **Other** "
                    "(still analyzed). Each metric belongs to at most one role."
                )

                property_candidates = candidate_user_property_columns(df, metrics)
                st.subheader("User properties (optional)")
                with st.container():
                    st.markdown(
                        '<div class="role-marker role-marker--property"></div>',
                        unsafe_allow_html=True,
                    )
                    st.multiselect(
                        "Segment dimensions",
                        options=property_candidates,
                        key="exp_selected_properties",
                        help="Optional columns (e.g. country, platform). After you run analysis, "
                        "filter cohorts in Step 3 without re-uploading the CSV.",
                    )

                run = st.button("Run analysis", type="primary", key="run_csv")

                if run:
                    cfg_run = st.session_state[CSV_METRICS_CONFIG_KEY]
                    roles_map = assign_roles(
                        metrics,
                        cfg_run["primary"],
                        cfg_run["secondary"],
                        cfg_run["guardrail"],
                    )
                    analyses = run_metric_analyses(
                        df, metrics, roles_map, control, treatment
                    )
                    st.session_state[EXP_BASE_DF_KEY] = df.copy()
                    st.session_state[EXP_WORK_DF_KEY] = df.copy()
                    st.session_state[EXP_ANALYSES_KEY] = analyses
                    st.session_state[EXP_METRICS_SNAP_KEY] = list(metrics)
                    st.session_state[EXP_CONTROL_SNAP_KEY] = control
                    st.session_state[EXP_TREATMENT_SNAP_KEY] = treatment
                    st.session_state[EXP_RESULTS_READY_KEY] = True
                    base_snapshot = st.session_state[EXP_BASE_DF_KEY]
                    for prop in st.session_state.get("exp_selected_properties") or []:
                        if prop not in base_snapshot.columns:
                            continue
                        wk = segment_filter_widget_key(prop)
                        if wk not in st.session_state:
                            st.session_state[wk] = property_uniques_for_segment(
                                base_snapshot, prop
                            )

                if st.session_state.get(EXP_RESULTS_READY_KEY):
                    base_df = st.session_state[EXP_BASE_DF_KEY]
                    for prop in st.session_state.get("exp_selected_properties") or []:
                        if prop not in base_df.columns:
                            continue
                        wk = segment_filter_widget_key(prop)
                        if wk not in st.session_state:
                            st.session_state[wk] = property_uniques_for_segment(
                                base_df, prop
                            )

                    if st.session_state.pop(EXP_SEGMENT_REAPPLY_KEY, False):
                        cfg_run = st.session_state[CSV_METRICS_CONFIG_KEY]
                        ctrl_snap = st.session_state[EXP_CONTROL_SNAP_KEY]
                        trt_snap = st.session_state[EXP_TREATMENT_SNAP_KEY]
                        metrics_snap = st.session_state[EXP_METRICS_SNAP_KEY]
                        sel_props = st.session_state.get("exp_selected_properties") or []
                        work_df = base_df.copy()
                        for prop in sel_props:
                            if prop not in work_df.columns:
                                continue
                            vals = st.session_state.get(
                                segment_filter_widget_key(prop), []
                            )
                            if vals:
                                work_df = work_df[work_df[prop].isin(vals)]
                        roles_map = assign_roles(
                            metrics_snap,
                            cfg_run["primary"],
                            cfg_run["secondary"],
                            cfg_run["guardrail"],
                        )
                        analyses = run_metric_analyses(
                            work_df,
                            metrics_snap,
                            roles_map,
                            ctrl_snap,
                            trt_snap,
                        )
                        st.session_state[EXP_WORK_DF_KEY] = work_df
                        st.session_state[EXP_ANALYSES_KEY] = analyses

                    cfg_run = st.session_state[CSV_METRICS_CONFIG_KEY]
                    primary = cfg_run["primary"]
                    ctrl = st.session_state[EXP_CONTROL_SNAP_KEY]
                    trt = st.session_state[EXP_TREATMENT_SNAP_KEY]
                    work_df = st.session_state[EXP_WORK_DF_KEY]
                    analyses = st.session_state[EXP_ANALYSES_KEY]

                    by_metric = {a.metric: a for a in analyses}
                    primary_ma = by_metric.get(primary)

                    st.header("3. Results")
                    st.markdown("#### Goal metrics")
                    st.caption("Experiment performance overview")
                    n_c = int((work_df["variant"] == ctrl).sum())
                    n_t = int((work_df["variant"] == trt).sum())
                    _vb = compute_variant_balance_for_srm(work_df, ctrl, trt)
                    if (
                        (_vb["n_c"] + _vb["n_t"]) > 0
                        and _vb["dev_pp"] >= HC_SRM_THRESHOLD_PP
                    ):
                        _pc = _vb["p_c"] * 100.0
                        _pt = _vb["p_t"] * 100.0
                        st.markdown(
                            '<div class="goal-metrics-variant-imbalance-alert">'
                            f"<strong>⚠️ Variant distribution is imbalanced:</strong><br>"
                            f"{html.escape(str(ctrl))} = {_pc:.1f}% · "
                            f"{html.escape(str(trt))} = {_pt:.1f}%<br>"
                            "<span>This may bias experiment results.</span>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    components.html(
                        build_goal_kpi_strip_fragment(
                            ctrl, trt, n_c, n_t, primary, primary_ma
                        ),
                        height=178,
                        scrolling=False,
                    )

                    st.markdown(
                        '<div class="segment-filters-card">',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h4>Segment filters</h4>", unsafe_allow_html=True)
                    sel_props = st.session_state.get("exp_selected_properties") or []
                    if not sel_props:
                        st.caption(
                            "Select columns under **User properties (optional)** in Step 2 "
                            "to filter cohorts here."
                        )
                    for prop in sel_props:
                        if prop not in base_df.columns:
                            continue
                        st.markdown(
                            f'<p class="segment-filter-label">'
                            f"{html.escape(str(prop))}</p>",
                            unsafe_allow_html=True,
                        )
                        opts = property_uniques_for_segment(base_df, prop)
                        wk = segment_filter_widget_key(prop)
                        with st.container():
                            st.markdown(
                                '<div class="role-marker role-marker--property"></div>',
                                unsafe_allow_html=True,
                            )
                            st.multiselect(
                                "values",
                                options=opts,
                                key=wk,
                                label_visibility="collapsed",
                            )
                    st.button(
                        "Update results",
                        on_click=exp_mark_segment_reapply,
                        key="exp_seg_apply_btn",
                    )
                    n_f = len(work_df)
                    n_ctrl_f = int((work_df["variant"] == ctrl).sum())
                    n_trt_f = int((work_df["variant"] == trt).sum())
                    st.markdown(
                        f'<p class="segment-filter-users">Filtered users: '
                        f"<strong>{n_f:,}</strong> "
                        f"(<strong>{html.escape(str(ctrl))}</strong>: {n_ctrl_f:,} · "
                        f"<strong>{html.escape(str(trt))}</strong>: {n_trt_f:,})"
                        f"</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if EXP_RESULTS_VIEW_KEY not in st.session_state:
                        st.session_state[EXP_RESULTS_VIEW_KEY] = RESULTS_VIEW_LABEL

                    st.caption(
                        "Switch between goal-metrics results and cohort health (balance checks)."
                    )
                    with st.container():
                        st.markdown(
                            '<span class="results-view-segmented-wrap"></span>',
                            unsafe_allow_html=True,
                        )
                        st.radio(
                            "View",
                            [RESULTS_VIEW_LABEL, HEALTH_CHECK_VIEW_LABEL],
                            horizontal=True,
                            label_visibility="collapsed",
                            key=EXP_RESULTS_VIEW_KEY,
                        )
                    view_sel = st.session_state[EXP_RESULTS_VIEW_KEY]

                    if view_sel == HEALTH_CHECK_VIEW_LABEL:
                        render_health_check_view(work_df, ctrl, trt, sel_props)
                    elif n_f == 0:
                        st.warning("No users match selected filters.")
                    else:
                        panel_html = build_goal_metrics_panel_html(
                            analyses, work_df, ctrl, trt, alpha
                        )
                        panel_h = min(1400, 120 + max(len(analyses), 1) * 92)
                        components.html(panel_html, height=panel_h, scrolling=True)

                    if view_sel == RESULTS_VIEW_LABEL and n_f > 0:
                        col_ctrl = f"Value ({ctrl})"
                        col_trt = f"Value ({trt})"
                        rows = []
                        for a in sort_analyses_by_role(analyses):
                            rows.append(
                                {
                                    "Metric": a.metric,
                                    "Role": a.role,
                                    "Kind": a.kind,
                                    col_ctrl: f"{a.value_control:.4f}",
                                    col_trt: f"{a.value_treatment:.4f}",
                                    "Uplift (rel.)": a.uplift,
                                    "p-value": a.p_value,
                                    "Probability": a.probability,
                                }
                            )
                        out_df = pd.DataFrame(rows)
                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download metrics (CSV)",
                            data=csv_bytes,
                            file_name="experiment_metrics_results.csv",
                            mime="text/csv",
                        )

                        with st.expander("Detailed numeric table"):
                            st.dataframe(
                                out_df.style.format(
                                    {
                                        "Uplift (rel.)": "{:.2%}",
                                        "p-value": "{:.4f}",
                                        "Probability": "{:.3f}",
                                    }
                                ),
                                use_container_width=True,
                            )

