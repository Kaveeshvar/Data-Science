"""
Auto Dashboard Generator v2
============================
Generates a professional HTML dashboard from CSV / XLSX files using Plotly.
Optionally uses an LLM (Groq) to pick chart specs.

Architecture:
    1. Load & coerce  ->  clean DataFrame
    2. Profile         ->  column roles + summary dict
    3. Spec            ->  LLM or rule-based chart/KPI spec
    4. Validate        ->  pre-chart checks (skip bad specs early)
    5. Aggregate       ->  produce ready-to-plot DataFrames
    6. Render charts   ->  Plotly figures with consistent theme
    7. Build HTML      ->  single-page dashboard
"""

import json
import logging
import os
import re
import html as html_module
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_COLS_PROFILE = 120
MAX_EXAMPLE_VALUES = 3
MAX_CHARTS = 6
MAX_KPIS = 8

MAX_POINTS_SCATTER = 10_000
MAX_POINTS_LINE = 20_000
TOP_N_CAT = 20

MAX_UNIQUE_CAT_X = 100_000       # skip categorical x with more uniques
MAX_NULL_PCT_REQUIRED = 80.0      # skip column if >80 % null

LLM_MAX_RETRIES = 2
LLM_RETRY_SLEEP = 1.0

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Consistent Plotly template applied to every figure
CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
CHART_PAPER_BG = "rgba(0,0,0,0)"
CHART_FONT = dict(family="Inter, system-ui, sans-serif", size=13, color="#c9d1d9")
CHART_MARGIN = dict(l=52, r=24, t=52, b=48)
CHART_COLORWAY = [
    "#6366f1", "#22d3ee", "#f472b6", "#facc15",
    "#34d399", "#fb923c", "#a78bfa", "#f87171",
]

# ╔══════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. FILE LOADING                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_table(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1")
    elif ext == ".xlsx":
        df = pd.read_excel(p, engine="openpyxl", sheet_name=sheet or 0)
    elif ext == ".xls":
        df = pd.read_excel(p, sheet_name=sheet or 0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    log.info("Loaded %s  (%d rows × %d cols)", p.name, *df.shape)
    return df

# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. DATA PROCESSING  (coerce + clean)                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def coerce_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convert object columns that look numeric (with commas / currency / %) to float."""
    df = df.copy()
    converted: List[str] = []
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        sample = df[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        cleaned = sample.str.replace(r"[,\s₹$€£¥]", "", regex=True).str.replace("%", "", regex=False)
        parse_rate = pd.to_numeric(cleaned, errors="coerce").notna().mean()
        if parse_rate >= 0.85:
            full = df[col].astype(str).str.replace(r"[,\s₹$€£¥]", "", regex=True).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(full, errors="coerce")
            converted.append(col)
            log.info("  Coerced to numeric: %s (%.0f%% parseable)", col, parse_rate * 100)
    return df, converted


def coerce_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Detect and convert object columns that look like dates/timestamps."""
    df = df.copy()
    parsed: List[str] = []
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        looks_date = sample.str.contains(r"\d{4}[-/]|[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True).mean()
        if looks_date < 0.6:
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            success_rate = dt.notna().mean()
            if success_rate >= 0.5:
                df[col] = dt
                parsed.append(col)
                log.info("  Parsed datetime: %s (%.0f%% success)", col, success_rate * 100)
        except Exception:
            log.warning("  Datetime parse failed for column: %s", col)
    return df, parsed


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: strip column names, coerce types."""
    df.columns = [str(c).strip() for c in df.columns]
    df, num_cols = coerce_numeric_columns(df)
    df, dt_cols = coerce_datetime_columns(df)
    return df

# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. COLUMN ROLE DETECTION                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [
        c for c in df.columns
        if (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype))
        and c not in datetime_cols
    ]
    return {"numeric": numeric, "datetime": datetime_cols, "categorical": categorical}

# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. LLM PROFILE + SPEC                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def dataframe_summary(df: pd.DataFrame) -> dict:
    summary: Dict[str, Any] = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": []}
    for col in df.columns[:MAX_COLS_PROFILE]:
        s = df[col]
        info: Dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "examples": s.dropna().astype(str).head(MAX_EXAMPLE_VALUES).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info["stats"] = {
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                "mean": float(np.nanmean(s)),
            }
        if pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            info["date_range"] = {"min": str(s.min()), "max": str(s.max())}
        summary["columns"].append(info)
    return summary


def _extract_json(text: str) -> str:
    text = text.strip()
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    raise ValueError("Could not extract JSON from model response.")


def _safe_json_loads(text: str) -> dict:
    raw = _extract_json(text)
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(raw)


def ask_groq_for_spec(summary: dict) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
Return STRICT JSON only. No markdown fences. No explanation.

Schema:
{{
  "title": "Dashboard Title",
  "kpis": [{{"label": "...", "type": "row_count|missing_pct|unique_count|sum|avg", "column": "optional"}}],
  "charts": [
    {{
      "kind": "histogram|bar|line|scatter|box",
      "title": "...",
      "x": "column_name",
      "y": "optional column_name",
      "color": "optional column_name",
      "agg": "optional: sum|avg|count",
      "top_n": 20
    }}
  ],
  "narrative": ["insight 1", "insight 2"]
}}

Rules:
- Only reference columns that exist in the summary.
- color must be a real categorical column (low cardinality).
- Prefer time trends when datetime columns exist.
- Avoid columns with very high n_unique (>50k) as x for bar charts.
- 3-6 charts total.  4-6 KPIs.

DATASET SUMMARY:
{json.dumps(summary)}
""".strip()

    last_err = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return _safe_json_loads(resp.choices[0].message.content.strip())
        except Exception as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP * (attempt + 1))
    raise last_err  # type: ignore[misc]


def default_spec(df: pd.DataFrame) -> dict:
    roles = get_column_roles(df)

    # --- KPIs ---
    kpis: List[dict] = [{"label": "Total Rows", "type": "row_count"}]
    for nc in roles["numeric"][:2]:
        kpis.append({"label": f"Avg {nc}", "type": "avg", "column": nc})
    for nc in roles["numeric"][:1]:
        kpis.append({"label": f"Sum {nc}", "type": "sum", "column": nc})
    for cc in roles["categorical"][:1]:
        kpis.append({"label": f"Unique {cc}", "type": "unique_count", "column": cc})

    # --- Charts ---
    charts: List[dict] = []

    # time + numeric -> line
    if roles["datetime"] and roles["numeric"]:
        charts.append({
            "kind": "line",
            "title": f"{roles['numeric'][0]} over time",
            "x": roles["datetime"][0],
            "y": roles["numeric"][0],
            "agg": "avg",
        })

    # cat + numeric -> bar
    if roles["categorical"] and roles["numeric"]:
        charts.append({
            "kind": "bar",
            "title": f"Avg {roles['numeric'][0]} by {roles['categorical'][0]}",
            "x": roles["categorical"][0],
            "y": roles["numeric"][0],
            "agg": "avg",
            "top_n": TOP_N_CAT,
        })

    # numeric distribution
    for nc in roles["numeric"][:1]:
        charts.append({"kind": "histogram", "title": f"Distribution of {nc}", "x": nc})

    # scatter numeric vs numeric
    if len(roles["numeric"]) >= 2:
        charts.append({
            "kind": "scatter",
            "title": f"{roles['numeric'][0]} vs {roles['numeric'][1]}",
            "x": roles["numeric"][0],
            "y": roles["numeric"][1],
        })

    # box plot  (cat × numeric)
    if roles["categorical"] and len(roles["numeric"]) >= 2:
        charts.append({
            "kind": "box",
            "title": f"{roles['numeric'][1]} by {roles['categorical'][0]}",
            "x": roles["categorical"][0],
            "y": roles["numeric"][1],
            "top_n": TOP_N_CAT,
        })

    # second bar if we have a second categorical
    if len(roles["categorical"]) >= 2 and roles["numeric"]:
        charts.append({
            "kind": "bar",
            "title": f"Avg {roles['numeric'][0]} by {roles['categorical'][1]}",
            "x": roles["categorical"][1],
            "y": roles["numeric"][0],
            "agg": "avg",
            "top_n": TOP_N_CAT,
        })

    return {
        "title": "Auto Dashboard",
        "kpis": kpis[:MAX_KPIS],
        "charts": charts[:MAX_CHARTS],
        "narrative": ["Dashboard auto-generated from detected column types and distributions."],
    }

# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. VALIDATION LAYER                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

class SkipChart(Exception):
    """Raised when a chart should be skipped with a reason."""


def validate_column(df: pd.DataFrame, col: str, *, required: bool = True) -> None:
    """Raise SkipChart if the column is missing or too null."""
    if col is None:
        raise SkipChart("Required column not specified")
    if col not in df.columns:
        raise SkipChart(f"Column '{col}' not found")
    if required:
        null_pct = df[col].isna().mean() * 100
        if null_pct > MAX_NULL_PCT_REQUIRED:
            raise SkipChart(f"Column '{col}' is {null_pct:.0f}% null")


def validate_categorical_cardinality(df: pd.DataFrame, col: str) -> None:
    """Skip if a categorical x has insane cardinality."""
    if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
        nuniq = df[col].nunique(dropna=True)
        if nuniq > MAX_UNIQUE_CAT_X:
            raise SkipChart(f"Column '{col}' has {nuniq:,} unique values — too many for categorical axis")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. AGGREGATION ENGINE                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def _top_n_filter(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Keep only the top-n most frequent values of *col*."""
    if col not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    top_vals = df[col].value_counts(dropna=True).head(n).index
    return df[df[col].isin(top_vals)]


def aggregate_bar(
    df: pd.DataFrame,
    x: str,
    y: Optional[str],
    color: Optional[str],
    agg: str = "avg",
    top_n: int = TOP_N_CAT,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Aggregate for a bar chart.
    Returns (agg_df, y_col_name, color_col_name_or_None).
    Raises SkipChart on problems.
    """
    validate_column(df, x)
    validate_categorical_cardinality(df, x)
    agg = (agg or "count").lower()

    # Resolve color
    safe_color: Optional[str] = None
    if color and color in df.columns and color != x:
        safe_color = color

    # Select needed columns
    cols = [x]
    if y and y in df.columns:
        cols.append(y)
    if safe_color:
        cols.append(safe_color)
    cols = list(dict.fromkeys(cols))  # dedupe, preserve order

    d = df[cols].dropna(subset=[x]).copy()
    if d.empty:
        raise SkipChart("All x-values are null")

    # Filter to top-n categories on x (and color)
    d = _top_n_filter(d, x, top_n)
    if safe_color:
        d = _top_n_filter(d, safe_color, min(top_n, 12))

    # Group-by columns
    group = [x] + ([safe_color] if safe_color else [])

    # Aggregate
    if y and y in d.columns and pd.api.types.is_numeric_dtype(d[y]):
        agg_map = {"sum": "sum", "avg": "mean", "mean": "mean", "count": "count"}
        agg_func = agg_map.get(agg, "mean")
        out = d.groupby(group, dropna=True)[y].agg(agg_func).reset_index()
        y_out = y
    else:
        out = d.groupby(group, dropna=True).size().reset_index(name="count")
        y_out = "count"

    if out.empty:
        raise SkipChart("Aggregation produced empty result")

    # Sort when no color split
    if not safe_color:
        out = out.sort_values(y_out, ascending=False).head(top_n)

    return out, y_out, safe_color


def aggregate_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
    agg: str = "avg",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Time-resample for a line chart.
    Returns (resampled_df, color_col_or_None).
    Raises SkipChart on problems.
    """
    validate_column(df, x)
    validate_column(df, y)

    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        raise SkipChart(f"x='{x}' is not datetime — cannot build line chart")
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise SkipChart(f"y='{y}' is not numeric — cannot build line chart")

    safe_color: Optional[str] = None
    if color and color in df.columns and color != x and color != y:
        safe_color = color

    cols = [x, y] + ([safe_color] if safe_color else [])
    d = df[cols].dropna(subset=[x, y]).copy()
    if d.empty:
        raise SkipChart("No valid rows after dropping nulls")

    d = d.sort_values(x)

    # Auto-detect resample freq
    span_days = max((d[x].max() - d[x].min()).days, 1)
    if span_days > 365 * 3:
        freq = "MS"   # month-start (avoids MonthEnd offset issues)
    elif span_days > 365:
        freq = "W"
    elif span_days > 60:
        freq = "W"
    elif span_days > 7:
        freq = "D"
    else:
        freq = "h"  # hourly for very short ranges

    agg = (agg or "avg").lower()
    agg_func = {"sum": "sum", "count": "count"}.get(agg, "mean")

    d = d.set_index(x)

    if safe_color:
        # Top-n color groups to keep chart readable
        top_groups = d[safe_color].value_counts(dropna=True).head(8).index
        d = d[d[safe_color].isin(top_groups)]
        parts = []
        for key, g in d.groupby(safe_color):
            r = g.resample(freq)[y].agg(agg_func).reset_index()
            r[safe_color] = key
            parts.append(r)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        out = d.resample(freq)[y].agg(agg_func).reset_index()

    if out.empty:
        raise SkipChart("Time aggregation produced empty result")

    if len(out) > MAX_POINTS_LINE:
        out = out.tail(MAX_POINTS_LINE)

    return out, safe_color


def prepare_histogram(
    df: pd.DataFrame,
    x: str,
    color: Optional[str],
    top_n: int = TOP_N_CAT,
) -> pd.DataFrame:
    """Return a (possibly sampled) DataFrame ready for px.histogram."""
    validate_column(df, x)
    cols = [x]
    if color and color in df.columns and color != x:
        cols.append(color)
    d = df[cols].dropna(subset=[x]).copy()
    if d.empty:
        raise SkipChart("No non-null values for histogram")

    # For non-numeric x, convert to value-count bar
    if not pd.api.types.is_numeric_dtype(d[x]):
        validate_categorical_cardinality(df, x)
        d = _top_n_filter(d, x, top_n)

    # Large datasets: subsample for histogram rendering (not aggregation)
    if pd.api.types.is_numeric_dtype(d[x]) and len(d) > 200_000:
        d = d.sample(200_000, random_state=42)

    return d


def prepare_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
) -> pd.DataFrame:
    """Return sampled-to-10k DataFrame ready for px.scatter."""
    validate_column(df, x)
    validate_column(df, y)
    cols = [x, y] + ([color] if (color and color in df.columns) else [])
    cols = list(dict.fromkeys(cols))
    d = df[cols].dropna(subset=[x, y]).copy()
    if d.empty:
        raise SkipChart("No valid rows for scatter")
    if len(d) > MAX_POINTS_SCATTER:
        d = d.sample(MAX_POINTS_SCATTER, random_state=42)
    return d


def prepare_box(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
    top_n: int = TOP_N_CAT,
) -> pd.DataFrame:
    """Return a filtered DataFrame ready for px.box."""
    validate_column(df, x)
    validate_column(df, y)
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise SkipChart(f"y='{y}' is not numeric for box plot")
    cols = [x, y] + ([color] if (color and color in df.columns and color != x) else [])
    cols = list(dict.fromkeys(cols))
    d = df[cols].dropna(subset=[x, y]).copy()
    if d.empty:
        raise SkipChart("No valid rows for box plot")

    d = _top_n_filter(d, x, top_n)
    if color and color in d.columns:
        d = _top_n_filter(d, color, min(top_n, 12))

    # Subsample for rendering if huge
    if len(d) > 200_000:
        d = d.sample(200_000, random_state=42)

    return d

# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. CHART RENDERER                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
    """Apply a consistent professional look to every figure."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        title=dict(text=title, font=dict(size=15, color="#e6edf3"), x=0.01, xanchor="left"),
        font=CHART_FONT,
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_BG,
        margin=CHART_MARGIN,
        colorway=CHART_COLORWAY,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        title_font=dict(size=12, color="#8b949e"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        title_font=dict(size=12, color="#8b949e"),
    )
    return fig


def _skipped_figure(reason: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠ {reason}",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6e7681"),
    )
    fig.update_layout(
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_BG,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    return fig


def build_chart(df: pd.DataFrame, spec: dict) -> go.Figure:
    """
    Build a single Plotly figure from a chart spec dict.
    Never raises — returns a skipped-figure on failure.
    """
    kind = (spec.get("kind") or "bar").lower()
    x = spec.get("x") or None
    y = spec.get("y") or None
    color = spec.get("color") or None
    agg = spec.get("agg", "avg")
    top_n = int(spec.get("top_n", TOP_N_CAT))
    title = spec.get("title") or kind.title()

    # Sanitize LLM returning literal string "None"
    if x and str(x).lower() == "none":
        x = None
    if y and str(y).lower() == "none":
        y = None
    if color and str(color).lower() == "none":
        color = None

    # Deduplicate: color/y must differ from x
    if color and color == x:
        color = None
    if y and y == x and kind not in ("histogram",):
        y = None

    # Resolve color safely
    if color and color not in df.columns:
        log.warning("Chart '%s': color column '%s' not found — ignoring", title, color)
        color = None

    try:
        # ---- HISTOGRAM ----
        if kind == "histogram":
            d = prepare_histogram(df, x, color, top_n)
            safe_color = color if (color and color in d.columns and color != x) else None
            if not pd.api.types.is_numeric_dtype(d[x]):
                # categorical: show count bar chart instead
                counts = d[x].value_counts().head(top_n).reset_index()
                counts.columns = [x, "count"]
                fig = px.bar(counts, x=x, y="count")
            else:
                fig = px.histogram(d, x=x, color=safe_color, nbins=50)
            return _apply_theme(fig, title)

        # ---- BAR ----
        if kind == "bar":
            out, y_col, safe_color = aggregate_bar(df, x, y, color, agg, top_n)
            fig = px.bar(out, x=x, y=y_col, color=safe_color, text_auto=".2s")
            fig.update_traces(textposition="outside", cliponaxis=False)
            return _apply_theme(fig, title)

        # ---- LINE ----
        if kind == "line":
            out, safe_color = aggregate_line(df, x, y, color, agg)
            fig = px.line(out, x=x, y=y, color=safe_color, markers=len(out) < 60)
            return _apply_theme(fig, title)

        # ---- SCATTER ----
        if kind == "scatter":
            d = prepare_scatter(df, x, y, color)
            safe_color = color if (color and color in d.columns) else None
            fig = px.scatter(d, x=x, y=y, color=safe_color, opacity=0.6)
            return _apply_theme(fig, title)

        # ---- BOX ----
        if kind == "box":
            d = prepare_box(df, x, y, color, top_n)
            safe_color = color if (color and color in d.columns and color != x) else None
            fig = px.box(d, x=x, y=y, color=safe_color)
            return _apply_theme(fig, title)

        raise SkipChart(f"Unsupported chart type: {kind}")

    except SkipChart as e:
        log.warning("SKIP chart '%s': %s", title, e)
        return _skipped_figure(str(e))
    except Exception as e:
        log.error("ERROR chart '%s': %s", title, e, exc_info=False)
        return _skipped_figure(str(e)[:120])

# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. KPI COMPUTATION                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_kpi(df: pd.DataFrame, kpi: dict) -> str:
    t = kpi.get("type", "")
    col = kpi.get("column")
    try:
        if t == "row_count":
            return f"{len(df):,}"
        if col and col not in df.columns:
            return "n/a"
        if t == "missing_pct" and col:
            return f"{df[col].isna().mean() * 100:.1f}%"
        if t == "unique_count" and col:
            return f"{df[col].nunique(dropna=True):,}"
        if t == "sum" and col and pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].sum(skipna=True)
            return _fmt_number(v)
        if t == "avg" and col and pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].mean(skipna=True)
            return _fmt_number(v)
    except Exception:
        pass
    return "n/a"


def _fmt_number(v: float) -> str:
    """Human-friendly number format."""
    abs_v = abs(v)
    if abs_v >= 1_000_000_000:
        return f"{v / 1_000_000_000:,.2f}B"
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:,.2f}M"
    if abs_v >= 1_000:
        return f"{v:,.1f}"
    return f"{v:,.2f}"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  9. DATA QUALITY                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_quality(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    total = max(rows * cols, 1)
    missing = int(df.isna().sum().sum())
    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing,
        "missing_pct": round(missing / total * 100, 2),
        "dupes": int(df.duplicated().sum()) if rows else 0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 1),
    }

# ╔══════════════════════════════════════════════════════════════════╗
# ║  10. HTML RENDERER                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

_esc = html_module.escape


def render_html(
    title: str,
    kpis: List[dict],
    charts_html: List[str],
    narrative: List[str],
    quality: dict,
) -> str:
    safe_title = _esc(title)

    # --- KPI cards ---
    kpi_cards = ""
    for k in kpis:
        kpi_cards += f"""
        <div class="kpi-card">
          <span class="kpi-label">{_esc(str(k['label']))}</span>
          <span class="kpi-value">{_esc(str(k['value']))}</span>
        </div>"""

    # --- Charts: split into primary (first 2) and secondary ---
    primary = charts_html[:2]
    secondary = charts_html[2:]

    primary_block = "\n".join(f'<div class="chart-card">{c}</div>' for c in primary)
    secondary_block = "\n".join(f'<div class="chart-card">{c}</div>' for c in secondary)

    # --- Narrative ---
    narrative_items = "".join(f"<li>{_esc(str(n))}</li>" for n in (narrative or []))

    # --- Quality metrics ---
    q = quality

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{safe_title}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
/* ── Reset & base ───────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:15px;-webkit-font-smoothing:antialiased}}
body{{
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  background:#0d1117;color:#e6edf3;line-height:1.55;
}}

/* ── Layout ─────────────────────────────────────────── */
.page{{max-width:1280px;margin:0 auto;padding:32px 28px 64px}}

/* ── Header ─────────────────────────────────────────── */
.header{{
  border-bottom:1px solid #21262d;
  padding-bottom:20px;margin-bottom:28px;
}}
.header h1{{font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;margin-bottom:4px}}
.header .subtitle{{color:#8b949e;font-size:0.85rem}}

/* ── Section titles ─────────────────────────────────── */
.section-title{{
  font-size:0.8rem;font-weight:600;text-transform:uppercase;
  letter-spacing:0.06em;color:#8b949e;margin:32px 0 14px;
}}

/* ── KPI row ────────────────────────────────────────── */
.kpi-row{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:14px;
}}
.kpi-card{{
  background:#161b22;border:1px solid #21262d;border-radius:12px;
  padding:18px 20px;display:flex;flex-direction:column;gap:6px;
  transition:border-color .15s;
}}
.kpi-card:hover{{border-color:#30363d}}
.kpi-label{{font-size:0.72rem;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;color:#8b949e}}
.kpi-value{{font-size:1.55rem;font-weight:700;color:#e6edf3}}

/* ── Chart grid ─────────────────────────────────────── */
.chart-grid-2{{
  display:grid;grid-template-columns:repeat(2,1fr);gap:16px;
}}
@media(max-width:860px){{
  .chart-grid-2{{grid-template-columns:1fr}}
}}
.chart-card{{
  background:#161b22;border:1px solid #21262d;border-radius:14px;
  padding:12px 14px 8px;overflow:hidden;
  box-shadow:0 1px 3px rgba(0,0,0,.25);
  transition:border-color .15s,box-shadow .15s;
}}
.chart-card:hover{{border-color:#30363d;box-shadow:0 4px 12px rgba(0,0,0,.35)}}

/* ── Info panels ────────────────────────────────────── */
.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}
@media(max-width:700px){{.info-grid{{grid-template-columns:1fr}}}}

.info-panel{{
  background:#161b22;border:1px solid #21262d;border-radius:12px;
  padding:22px 24px;
}}
.info-panel h2{{font-size:0.95rem;font-weight:600;margin-bottom:14px;color:#e6edf3}}
.info-panel ul{{padding-left:18px;color:#c9d1d9;font-size:0.88rem;line-height:1.7}}
.info-panel ul li::marker{{color:#30363d}}

.quality-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;
}}
.quality-item{{display:flex;flex-direction:column;gap:2px}}
.quality-item .ql{{font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.04em}}
.quality-item .qv{{font-size:1.05rem;font-weight:600}}

/* ── Footer ─────────────────────────────────────────── */
.footer{{
  margin-top:48px;padding-top:18px;border-top:1px solid #21262d;
  text-align:center;color:#484f58;font-size:0.75rem;
}}
</style>
</head>
<body>
<div class="page">

  <!-- ═══ HEADER ═══ -->
  <div class="header">
    <h1>{safe_title}</h1>
    <div class="subtitle">Auto-generated dashboard &middot; {q['rows']:,} rows &middot; {q['cols']} columns</div>
  </div>

  <!-- ═══ KPI ROW ═══ -->
  <div class="section-title">Key Metrics</div>
  <div class="kpi-row">{kpi_cards}
  </div>

  <!-- ═══ PRIMARY CHARTS (2-col) ═══ -->
  <div class="section-title">Primary Analysis</div>
  <div class="chart-grid-2">
    {primary_block}
  </div>

  <!-- ═══ SECONDARY CHARTS (2-col) ═══ -->
  {"" if not secondary_block else f'''
  <div class="section-title">Additional Charts</div>
  <div class="chart-grid-2">
    {secondary_block}
  </div>'''}

  <!-- ═══ DATA QUALITY + NARRATIVE ═══ -->
  <div class="info-grid">
    <div class="info-panel">
      <h2>Data Quality</h2>
      <div class="quality-grid">
        <div class="quality-item"><span class="ql">Rows</span><span class="qv">{q['rows']:,}</span></div>
        <div class="quality-item"><span class="ql">Columns</span><span class="qv">{q['cols']}</span></div>
        <div class="quality-item"><span class="ql">Duplicates</span><span class="qv">{q['dupes']:,}</span></div>
        <div class="quality-item"><span class="ql">Missing Cells</span><span class="qv">{q['missing_cells']:,} ({q['missing_pct']:.1f}%)</span></div>
        <div class="quality-item"><span class="ql">Memory</span><span class="qv">{q['memory_mb']:.1f} MB</span></div>
      </div>
    </div>
    <div class="info-panel">
      <h2>Analyst Summary</h2>
      <ul>{narrative_items}</ul>
    </div>
  </div>

  <div class="footer">Generated by Auto Dashboard v2</div>
</div>
</body>
</html>"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  11. MAIN ORCHESTRATOR                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def _generate_html_filename(input_path: str) -> str:
    """Generate a custom HTML filename from the dataset name."""
    p = Path(input_path)
    stem = p.stem  # filename without extension (e.g., "banking_dataset")
    # Convert to title case and add "Dashboard" suffix
    friendly_name = stem.replace("_", " ").title()
    output_name = f"{stem}_dashboard.html"
    return output_name


def main(
    input_path: str,
    out_html: Optional[str] = None,
    sheet: Optional[str] = None,
    use_groq: bool = False,
):
    # Auto-generate HTML filename if not provided
    if out_html is None:
        out_html = _generate_html_filename(input_path)
    
    # ── Load & process ──
    df = load_table(input_path, sheet=sheet)
    df = process_dataframe(df)

    # ── Build spec ──
    summary = dataframe_summary(df)

    if use_groq:
        try:
            spec = ask_groq_for_spec(summary)
            log.info("LLM spec received: %d charts, %d KPIs", len(spec.get("charts", [])), len(spec.get("kpis", [])))
        except Exception as e:
            log.warning("Groq spec failed (%s). Using default.", e)
            spec = default_spec(df)
    else:
        spec = default_spec(df)

    # ── KPIs ──
    kpis = []
    for k in (spec.get("kpis") or [])[:MAX_KPIS]:
        kpis.append({"label": k.get("label", "KPI"), "value": compute_kpi(df, k)})
    if not kpis:
        kpis = [{"label": "Rows", "value": f"{len(df):,}"}]

    # ── Charts ──
    chart_specs = (spec.get("charts") or [])[:MAX_CHARTS]
    if not chart_specs:
        chart_specs = default_spec(df).get("charts", [])

    charts_html: List[str] = []
    for cs in chart_specs:
        fig = build_chart(df, cs)
        charts_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ── Assemble HTML ──
    quality = compute_quality(df)
    html_str = render_html(
        title=str(spec.get("title", "Dashboard")),
        kpis=kpis,
        charts_html=charts_html,
        narrative=spec.get("narrative") or [],
        quality=quality,
    )
    Path(out_html).write_text(html_str, encoding="utf-8")
    log.info("Dashboard saved → %s", out_html)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CLI                                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a professional HTML dashboard from CSV/XLSX.")
    parser.add_argument("input", nargs="?", default=None, help="Path to .csv / .xlsx / .xls file (optional)")
    parser.add_argument("--out", default=None, help="Output HTML path (auto-generated if not specified)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--groq", action="store_true", help="Use Groq LLM for chart spec")
    args = parser.parse_args()

    # If no input file provided, prompt for it
    input_file = args.input
    if not input_file:
        input_file = input("\n📊 Enter path to CSV/XLSX file: ").strip()
        if not input_file:
            print("❌ No file provided. Exiting.")
            exit(1)

    main(input_file, out_html=args.out, sheet=args.sheet, use_groq=args.groq)
