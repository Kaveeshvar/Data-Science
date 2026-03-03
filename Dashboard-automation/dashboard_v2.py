"""
Pro Dashboard Generator v2
============================
Production-grade HTML dashboard generator from CSV / XLSX files.

Features
--------
- 6 professional themes auto-selected by dataset domain
- Outlier detection & visualization (IQR + Z-score)
- Correlation matrix & feature-importance analysis
- ML model recommendations with reasoning
- Domain-specific AI insights (business / medical / technical / …)
- Multiple targeted LLM calls for rich analysis
- Professional Plotly charts with consistent per-theme styling

Architecture
------------
 1. Load & coerce   → clean DataFrame
 2. Profile         → column roles + summary dict
 3. Domain detect   → LLM or rule-based domain classification
 4. Outlier analysis→ IQR / Z-score outlier detection
 5. Correlation     → Pearson matrix + top pairs + feature importance
 6. Chart spec      → LLM or rule-based chart / KPI spec
 7. ML recommend    → model suggestions with reasoning
 8. Domain insights → LLM-generated production insights
 9. Template select → choose theme based on domain
10. Validate / Aggregate / Render charts
11. Assemble single-page HTML dashboard
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
#   CONFIG
# ═══════════════════════════════════════════════════════════════════

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FAST_MODEL: str = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

MAX_COLS_PROFILE   = 120
MAX_EXAMPLE_VALUES = 4
MAX_CHARTS         = 8
MAX_KPIS           = 8

MAX_POINTS_SCATTER = 10_000
MAX_POINTS_LINE    = 20_000
TOP_N_CAT          = 20

MAX_UNIQUE_CAT_X      = 100_000
MAX_NULL_PCT_REQUIRED  = 80.0

LLM_MAX_RETRIES  = 2
LLM_RETRY_SLEEP  = 1.5
LLM_CALL_DELAY   = 0.8          # seconds between consecutive API calls

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

BASE_DIR    = Path(__file__).resolve().parent
INPUTS_DIR  = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ═══════════════════════════════════════════════════════════════════
#   THEMES  (6 professional palettes)
# ═══════════════════════════════════════════════════════════════════

THEMES: Dict[str, Dict[str, Any]] = {
    # ── 1. Midnight Pro  (dark / tech / default) ────────────────
    "midnight": {
        "name": "Midnight Pro",
        "font_family": "'Inter','SF Pro Display',system-ui,-apple-system,sans-serif",
        "body_bg": "#0d1117",
        "card_bg": "#161b22",
        "card_bg_alt": "#1c2128",
        "card_border": "#21262d",
        "card_border_hover": "#388bfd44",
        "text_primary": "#e6edf3",
        "text_secondary": "#8b949e",
        "text_muted": "#484f58",
        "accent_primary": "#818cf8",
        "accent_secondary": "#22d3ee",
        "accent_success": "#34d399",
        "accent_warning": "#facc15",
        "accent_danger": "#f87171",
        "gradient_start": "#6366f1",
        "gradient_end": "#a78bfa",
        "kpi_icon_bg": "#6366f120",
        "shadow": "0 1px 3px rgba(0,0,0,.3),0 1px 2px rgba(0,0,0,.2)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.45)",
        "plotly_template": "plotly_dark",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(255,255,255,.06)",
        "chart_colorway": [
            "#818cf8", "#22d3ee", "#f472b6", "#facc15",
            "#34d399", "#fb923c", "#a78bfa", "#f87171",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
    # ── 2. Clinical White  (light / medical / healthcare) ──────
    "clinical": {
        "name": "Clinical White",
        "font_family": "'Inter','SF Pro Display',system-ui,-apple-system,sans-serif",
        "body_bg": "#f8fafc",
        "card_bg": "#ffffff",
        "card_bg_alt": "#f1f5f9",
        "card_border": "#e2e8f0",
        "card_border_hover": "#0891b230",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_muted": "#94a3b8",
        "accent_primary": "#0891b2",
        "accent_secondary": "#059669",
        "accent_success": "#16a34a",
        "accent_warning": "#d97706",
        "accent_danger": "#dc2626",
        "gradient_start": "#0891b2",
        "gradient_end": "#0ea5e9",
        "kpi_icon_bg": "#0891b218",
        "shadow": "0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.12)",
        "plotly_template": "plotly_white",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(0,0,0,.07)",
        "chart_colorway": [
            "#0891b2", "#059669", "#7c3aed", "#ea580c",
            "#2563eb", "#d946ef", "#0d9488", "#dc2626",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
    # ── 3. Ocean Corporate  (dark blue / finance / business) ───
    "ocean": {
        "name": "Ocean Corporate",
        "font_family": "'Inter','Segoe UI',system-ui,sans-serif",
        "body_bg": "#0f172a",
        "card_bg": "#1e293b",
        "card_bg_alt": "#1a2332",
        "card_border": "#334155",
        "card_border_hover": "#3b82f640",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "text_muted": "#64748b",
        "accent_primary": "#60a5fa",
        "accent_secondary": "#fbbf24",
        "accent_success": "#4ade80",
        "accent_warning": "#fbbf24",
        "accent_danger": "#f87171",
        "gradient_start": "#1d4ed8",
        "gradient_end": "#3b82f6",
        "kpi_icon_bg": "#3b82f620",
        "shadow": "0 1px 3px rgba(0,0,0,.3),0 1px 2px rgba(0,0,0,.2)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.45)",
        "plotly_template": "plotly_dark",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(255,255,255,.06)",
        "chart_colorway": [
            "#60a5fa", "#fbbf24", "#34d399", "#f472b6",
            "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
    # ── 4. Sunset Vibrant  (warm dark / marketing / sales) ─────
    "sunset": {
        "name": "Sunset Vibrant",
        "font_family": "'Inter',system-ui,sans-serif",
        "body_bg": "#18181b",
        "card_bg": "#27272a",
        "card_bg_alt": "#2d2d30",
        "card_border": "#3f3f46",
        "card_border_hover": "#f9731650",
        "text_primary": "#fafafa",
        "text_secondary": "#a1a1aa",
        "text_muted": "#71717a",
        "accent_primary": "#f97316",
        "accent_secondary": "#ec4899",
        "accent_success": "#22c55e",
        "accent_warning": "#eab308",
        "accent_danger": "#ef4444",
        "gradient_start": "#ea580c",
        "gradient_end": "#f97316",
        "kpi_icon_bg": "#f9731620",
        "shadow": "0 1px 3px rgba(0,0,0,.3),0 1px 2px rgba(0,0,0,.2)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.45)",
        "plotly_template": "plotly_dark",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(255,255,255,.06)",
        "chart_colorway": [
            "#fb923c", "#f472b6", "#a78bfa", "#facc15",
            "#34d399", "#38bdf8", "#e879f9", "#f87171",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
    # ── 5. Forest Nature  (dark green / environmental / bio) ───
    "forest": {
        "name": "Forest Nature",
        "font_family": "'Inter',system-ui,sans-serif",
        "body_bg": "#052e16",
        "card_bg": "#14532d",
        "card_bg_alt": "#166534",
        "card_border": "#16a34a",
        "card_border_hover": "#4ade8040",
        "text_primary": "#f0fdf4",
        "text_secondary": "#86efac",
        "text_muted": "#4ade80",
        "accent_primary": "#4ade80",
        "accent_secondary": "#fbbf24",
        "accent_success": "#86efac",
        "accent_warning": "#fbbf24",
        "accent_danger": "#fca5a5",
        "gradient_start": "#15803d",
        "gradient_end": "#22c55e",
        "kpi_icon_bg": "#4ade8020",
        "shadow": "0 1px 3px rgba(0,0,0,.35),0 1px 2px rgba(0,0,0,.25)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.5)",
        "plotly_template": "plotly_dark",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(255,255,255,.08)",
        "chart_colorway": [
            "#4ade80", "#fbbf24", "#60a5fa", "#f472b6",
            "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
    # ── 6. Slate Modern  (grey / general / academic) ───────────
    "slate": {
        "name": "Slate Modern",
        "font_family": "'Inter',system-ui,sans-serif",
        "body_bg": "#1e293b",
        "card_bg": "#334155",
        "card_bg_alt": "#2d3a4d",
        "card_border": "#475569",
        "card_border_hover": "#8b5cf640",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#64748b",
        "accent_primary": "#a78bfa",
        "accent_secondary": "#22d3ee",
        "accent_success": "#34d399",
        "accent_warning": "#fbbf24",
        "accent_danger": "#f87171",
        "gradient_start": "#7c3aed",
        "gradient_end": "#a78bfa",
        "kpi_icon_bg": "#8b5cf620",
        "shadow": "0 1px 3px rgba(0,0,0,.25),0 1px 2px rgba(0,0,0,.15)",
        "shadow_hover": "0 8px 24px rgba(0,0,0,.35)",
        "plotly_template": "plotly_dark",
        "chart_bg": "rgba(0,0,0,0)",
        "chart_paper_bg": "rgba(0,0,0,0)",
        "chart_grid": "rgba(255,255,255,.06)",
        "chart_colorway": [
            "#a78bfa", "#22d3ee", "#f472b6", "#fbbf24",
            "#34d399", "#fb923c", "#818cf8", "#f87171",
        ],
        "heatmap_colorscale": "RdBu_r",
    },
}

# domain  →  theme
DOMAIN_THEME_MAP: Dict[str, str] = {
    "business":      "ocean",
    "finance":       "ocean",
    "banking":       "ocean",
    "medical":       "clinical",
    "healthcare":    "clinical",
    "clinical":      "clinical",
    "marketing":     "sunset",
    "sales":         "sunset",
    "ecommerce":     "sunset",
    "technical":     "midnight",
    "engineering":   "midnight",
    "iot":           "midnight",
    "academic":      "slate",
    "research":      "slate",
    "education":     "slate",
    "environmental": "forest",
    "agriculture":   "forest",
    "biology":       "forest",
    "general":       "midnight",
}

# keywords for rule-based domain detection
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance":       ["revenue", "profit", "price", "stock", "portfolio", "interest",
                      "loan", "credit", "debit", "balance", "transaction", "payment",
                      "income", "budget", "roi", "margin", "turnover", "expense", "cost",
                      "dividend", "equity", "asset", "liability"],
    "medical":       ["patient", "diagnosis", "blood", "bmi", "heart", "disease",
                      "treatment", "hospital", "health", "clinical", "symptom",
                      "medicine", "prescription", "pulse", "cholesterol", "glucose",
                      "hemoglobin", "platelet", "bp", "oxygen"],
    "marketing":     ["campaign", "click", "impression", "conversion", "engagement",
                      "ad", "ctr", "bounce", "visitor", "session", "lead", "funnel",
                      "segment", "retention", "churn", "lifetime"],
    "sales":         ["customer", "order", "product", "quantity", "discount", "ship",
                      "purchase", "cart", "invoice", "sku"],
    "technical":     ["sensor", "temperature", "pressure", "voltage", "current", "rpm",
                      "error", "latency", "throughput", "bandwidth", "cpu", "memory",
                      "request", "response", "uptime"],
    "academic":      ["score", "grade", "gpa", "student", "exam", "course",
                      "enrollment", "attendance", "university", "admission"],
    "environmental": ["species", "habitat", "emission", "pollution", "climate",
                      "weather", "rainfall", "soil", "crop", "yield", "forest"],
}

# ═══════════════════════════════════════════════════════════════════
#   LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard_v2")

# ═══════════════════════════════════════════════════════════════════
#   1.  FILE LOADING
# ═══════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════
#   2.  DATA PROCESSING  (coerce + clean)
# ═══════════════════════════════════════════════════════════════════

def coerce_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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
    df = df.copy()
    parsed: List[str] = []
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        looks_date = sample.str.contains(
            r"\d{4}[-/]|[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True
        ).mean()
        if looks_date < 0.6:
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().mean() >= 0.5:
                df[col] = dt
                parsed.append(col)
                log.info("  Parsed datetime: %s", col)
        except Exception:
            pass
    return df, parsed


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df, _ = coerce_numeric_columns(df)
    df, _ = coerce_datetime_columns(df)
    return df

# ═══════════════════════════════════════════════════════════════════
#   3.  COLUMN ROLE DETECTION
# ═══════════════════════════════════════════════════════════════════

def get_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [
        c for c in df.columns
        if (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype))
        and c not in datetime_cols
    ]
    return {"numeric": numeric, "datetime": datetime_cols, "categorical": categorical}

# ═══════════════════════════════════════════════════════════════════
#   4.  DATA PROFILING  (summary for LLM)
# ═══════════════════════════════════════════════════════════════════

def dataframe_summary(df: pd.DataFrame) -> dict:
    summary: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
    }
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
                "mean": round(float(np.nanmean(s)), 4),
                "std": round(float(np.nanstd(s)), 4),
                "median": float(np.nanmedian(s)),
            }
        if pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            info["date_range"] = {"min": str(s.min()), "max": str(s.max())}
        summary["columns"].append(info)
    return summary

# ═══════════════════════════════════════════════════════════════════
#   5.  JSON HELPERS
# ═══════════════════════════════════════════════════════════════════

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
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)        # trailing commas
    raw = re.sub(r"[\x00-\x1f]", " ", raw)           # control chars
    return json.loads(raw)

# ═══════════════════════════════════════════════════════════════════
#   6.  LLM ENGINE  (multiple targeted calls)
# ═══════════════════════════════════════════════════════════════════

def _get_groq_client() -> Optional[Groq]:
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None


def _llm_call(
    client: Groq,
    prompt: str,
    system: str = "You output valid JSON only.",
    model: Optional[str] = None,
) -> dict:
    """Single LLM call with retry logic.  Returns parsed dict."""
    model = model or GROQ_MODEL
    last_err: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return _safe_json_loads(resp.choices[0].message.content.strip())
        except Exception as e:
            last_err = e
            log.warning("LLM attempt %d failed: %s", attempt + 1, e)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP * (attempt + 1))
    raise last_err  # type: ignore[misc]


# ── 6a.  Domain detection ──────────────────────────────────────

def detect_domain_llm(summary: dict, client: Groq) -> dict:
    prompt = f"""Analyze this dataset and classify its domain.
Return STRICT JSON only. No markdown fences.

Schema:
{{
  "domain": "one of: finance, medical, marketing, sales, technical, academic, environmental, general",
  "sub_domain": "more specific area (e.g., credit risk, patient outcomes, ad performance)",
  "reasoning": "one sentence why"
}}

DATASET SUMMARY:
{json.dumps(summary, default=str)}"""
    try:
        return _llm_call(client, prompt, model=GROQ_FAST_MODEL)
    except Exception as e:
        log.warning("Domain detection LLM failed: %s", e)
        return {"domain": "general", "sub_domain": "unknown", "reasoning": "LLM call failed"}


def detect_domain_rules(df: pd.DataFrame) -> dict:
    """Rule-based domain detection from column names and sample values."""
    all_text = " ".join(str(c).lower() for c in df.columns)
    # add sample values
    for col in df.columns[:20]:
        vals = df[col].dropna().astype(str).head(10).tolist()
        all_text += " " + " ".join(v.lower() for v in vals)

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in all_text)

    if not scores or max(scores.values()) == 0:
        return {"domain": "general", "sub_domain": "unknown", "reasoning": "No domain keywords matched"}

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return {"domain": best, "sub_domain": best, "reasoning": f"Matched {scores[best]} domain keywords"}


# ── 6b.  Chart specification ───────────────────────────────────

def ask_groq_for_spec(summary: dict, client: Groq) -> dict:
    prompt = f"""Return STRICT JSON only. No markdown fences.

Schema:
{{
  "title": "Dashboard Title",
  "kpis": [{{"label": "...", "type": "row_count|missing_pct|unique_count|sum|avg|max|min", "column": "optional"}}],
  "charts": [
    {{
      "kind": "histogram|bar|line|scatter|box|pie",
      "title": "Descriptive chart title",
      "x": "column_name",
      "y": "optional column_name",
      "color": "optional categorical column_name",
      "agg": "optional: sum|avg|count",
      "top_n": 20
    }}
  ],
  "narrative": ["insight 1", "insight 2", "insight 3"]
}}

Rules:
- Only reference columns from the summary.
- color must be a REAL categorical column with low cardinality (<20 unique).
- Prefer time-trend lines when datetime columns exist.
- Avoid high-cardinality (>50k unique) columns as x for bar charts.
- 4–6 charts total.  4–8 KPIs.
- Make EVERY chart title specific and insightful (not generic).
- For bar charts, always specify agg (sum, avg, or count).

DATASET SUMMARY:
{json.dumps(summary, default=str)}"""
    return _llm_call(client, prompt)


# ── 6c.  ML model recommendation ──────────────────────────────

def ask_groq_for_ml(
    summary: dict,
    outlier_summary: str,
    corr_summary: str,
    client: Groq,
) -> dict:
    prompt = f"""You are a senior ML engineer. Given this dataset profile,
recommend the best machine learning approach.

Return STRICT JSON only. No markdown fences.

Schema:
{{
  "task_type": "classification|regression|clustering|time_series|anomaly_detection",
  "target_variable": "column_name or null",
  "target_reasoning": "why this column is the target",
  "models": [
    {{
      "rank": 1,
      "name": "Model Name",
      "suitability_score": 85,
      "reasoning": "2-3 sentences why this model fits",
      "pros": ["pro1", "pro2"],
      "cons": ["con1", "con2"]
    }}
  ],
  "preprocessing_steps": ["step 1", "step 2"],
  "warnings": ["potential issue"]
}}

Rules:
- Recommend exactly 3 models ranked by suitability.
- suitability_score is 0-100.
- Be specific about WHY each model is suited for THIS dataset.
- Consider dataset size, feature types, outliers, correlations.

DATASET SUMMARY:
{json.dumps(summary, default=str)}

OUTLIER ANALYSIS:
{outlier_summary}

CORRELATION ANALYSIS:
{corr_summary}"""
    return _llm_call(client, prompt)


# ── 6d.  Domain-specific insights ─────────────────────────────

def ask_groq_for_insights(
    summary: dict,
    domain: str,
    outlier_summary: str,
    corr_summary: str,
    ml_summary: str,
    client: Groq,
) -> list:
    domain_label = domain.replace("_", " ").title()
    prompt = f"""You are a senior {domain_label} data analyst.
Given this dataset, provide production-grade actionable insights.

Return STRICT JSON only. No markdown fences.

Schema:
{{
  "insights": [
    {{
      "title": "Short insight title",
      "detail": "Detailed explanation (2-3 sentences with specific numbers/percentages)",
      "impact": "high|medium|low",
      "category": "trend|risk|opportunity|pattern|anomaly|recommendation",
      "recommendation": "Specific action to take (1-2 sentences)"
    }}
  ]
}}

Rules:
- Provide exactly 5 insights.
- Each insight MUST reference specific columns or statistics from the data.
- Insights should be {domain_label}-relevant and actionable.
- Include at least one risk, one opportunity, and one pattern.
- Be specific with numbers (use stats from the summary).

DOMAIN: {domain_label}

DATASET SUMMARY:
{json.dumps(summary, default=str)}

OUTLIER ANALYSIS:
{outlier_summary}

CORRELATION FINDINGS:
{corr_summary}

ML RECOMMENDATION:
{ml_summary}"""
    try:
        result = _llm_call(client, prompt)
        return result.get("insights", [])
    except Exception as e:
        log.warning("Insights LLM failed: %s", e)
        return []

# ═══════════════════════════════════════════════════════════════════
#   7.  FALLBACK / RULE-BASED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def default_spec(df: pd.DataFrame) -> dict:
    roles = get_column_roles(df)
    kpis: List[dict] = [{"label": "Total Records", "type": "row_count"}]
    for nc in roles["numeric"][:3]:
        kpis.append({"label": f"Avg {nc}", "type": "avg", "column": nc})
    for nc in roles["numeric"][:1]:
        kpis.append({"label": f"Total {nc}", "type": "sum", "column": nc})
    for nc in roles["numeric"][:1]:
        kpis.append({"label": f"Max {nc}", "type": "max", "column": nc})
    for cc in roles["categorical"][:1]:
        kpis.append({"label": f"Unique {cc}", "type": "unique_count", "column": cc})

    charts: List[dict] = []
    if roles["datetime"] and roles["numeric"]:
        charts.append({
            "kind": "line", "title": f"{roles['numeric'][0]} Over Time",
            "x": roles["datetime"][0], "y": roles["numeric"][0], "agg": "avg",
        })
    if roles["categorical"] and roles["numeric"]:
        charts.append({
            "kind": "bar", "title": f"Avg {roles['numeric'][0]} by {roles['categorical'][0]}",
            "x": roles["categorical"][0], "y": roles["numeric"][0], "agg": "avg", "top_n": TOP_N_CAT,
        })
    for nc in roles["numeric"][:1]:
        charts.append({"kind": "histogram", "title": f"Distribution of {nc}", "x": nc})
    if len(roles["numeric"]) >= 2:
        charts.append({
            "kind": "scatter", "title": f"{roles['numeric'][0]} vs {roles['numeric'][1]}",
            "x": roles["numeric"][0], "y": roles["numeric"][1],
        })
    if roles["categorical"] and len(roles["numeric"]) >= 2:
        charts.append({
            "kind": "box", "title": f"{roles['numeric'][1]} by {roles['categorical'][0]}",
            "x": roles["categorical"][0], "y": roles["numeric"][1], "top_n": TOP_N_CAT,
        })
    if len(roles["categorical"]) >= 2 and roles["numeric"]:
        charts.append({
            "kind": "bar", "title": f"Avg {roles['numeric'][0]} by {roles['categorical'][1]}",
            "x": roles["categorical"][1], "y": roles["numeric"][0], "agg": "avg", "top_n": TOP_N_CAT,
        })
    # pie chart if suitable categorical exists
    if roles["categorical"]:
        for cc in roles["categorical"][:3]:
            if 2 <= df[cc].nunique() <= 8:
                charts.append({"kind": "pie", "title": f"Distribution of {cc}", "x": cc})
                break

    return {
        "title": "Data Analysis Dashboard",
        "kpis": kpis[:MAX_KPIS],
        "charts": charts[:MAX_CHARTS],
        "narrative": ["Dashboard auto-generated from detected column types.",
                       "Charts show distributions, relationships, and trends in the data."],
    }


def _identify_target(df: pd.DataFrame, roles: dict) -> Optional[str]:
    """Heuristic to identify a likely target / label column."""
    target_names = {
        "target", "label", "class", "y", "outcome", "result", "status",
        "churn", "default", "survived", "diagnosis", "price", "salary",
        "revenue", "amount", "approved", "fraud", "spam", "sentiment",
    }
    for col in df.columns:
        if col.lower().strip().replace("_", "").replace(" ", "") in target_names:
            return col
    # last column with low cardinality might be target
    last = df.columns[-1]
    if last in roles["categorical"] and df[last].nunique() <= 15:
        return last
    if last in roles["numeric"] and df[last].nunique() <= 15:
        return last
    return None


def fallback_ml_recommendation(df: pd.DataFrame, roles: dict) -> dict:
    n_rows = len(df)
    target = _identify_target(df, roles)

    if target is None:
        return {
            "task_type": "clustering",
            "target_variable": None,
            "target_reasoning": "No clear target variable detected; unsupervised learning recommended.",
            "models": [
                {"rank": 1, "name": "K-Means Clustering", "suitability_score": 80,
                 "reasoning": "Good baseline for discovering natural groupings in the data.",
                 "pros": ["Simple & fast", "Scales well", "Easy to interpret"],
                 "cons": ["Requires specifying K", "Assumes spherical clusters", "Sensitive to outliers"]},
                {"rank": 2, "name": "DBSCAN", "suitability_score": 72,
                 "reasoning": "Density-based clustering that handles arbitrary-shaped clusters and noise.",
                 "pros": ["No K required", "Finds arbitrary shapes", "Robust to outliers"],
                 "cons": ["Sensitive to eps/min_samples", "Struggles with varying density", "Slower on large data"]},
                {"rank": 3, "name": "Hierarchical (Agglomerative)", "suitability_score": 65,
                 "reasoning": "Creates a dendrogram for understanding cluster hierarchy.",
                 "pros": ["Dendrogram visualization", "No K required upfront", "Works well for small data"],
                 "cons": ["O(n³) complexity", "Cannot undo merges", "Memory intensive"]},
            ],
            "preprocessing_steps": [
                "Handle missing values (imputation or removal)",
                "Encode categorical variables (One-Hot or Label Encoding)",
                "Scale numeric features (StandardScaler or MinMaxScaler)",
                "Remove or cap outliers",
            ],
            "warnings": ["No target variable detected — results are exploratory."],
        }

    # Determine classification vs regression
    is_cat = target in roles["categorical"]
    n_unique = df[target].nunique()
    is_classification = is_cat or n_unique <= 20

    if is_classification:
        task = "classification"
        if n_rows < 5000:
            models = [
                {"rank": 1, "name": "Random Forest Classifier", "suitability_score": 85,
                 "reasoning": f"Strong baseline for {n_rows:,} rows. Handles mixed features and provides feature importance.",
                 "pros": ["Robust to overfitting", "Handles missing data implicitly", "Feature importance"],
                 "cons": ["Slower inference than linear models", "Less interpretable than single tree"]},
                {"rank": 2, "name": "Logistic Regression", "suitability_score": 78,
                 "reasoning": "Simple, interpretable, and fast. Good starting point for classification.",
                 "pros": ["Fast training/inference", "Highly interpretable", "Probabilistic output"],
                 "cons": ["Assumes linear decision boundary", "Requires feature scaling", "Struggles with non-linear patterns"]},
                {"rank": 3, "name": "Support Vector Classifier", "suitability_score": 72,
                 "reasoning": "Effective on smaller datasets with clear margins between classes.",
                 "pros": ["Effective in high dimensions", "Memory efficient (support vectors)", "Kernel trick for non-linear"],
                 "cons": ["Slow on large datasets", "Sensitive to feature scaling", "No probability estimates by default"]},
            ]
        elif n_rows < 100_000:
            models = [
                {"rank": 1, "name": "XGBoost Classifier", "suitability_score": 92,
                 "reasoning": f"Excellent for {n_rows:,} rows with gradient boosting. Handles missing data and mixed types.",
                 "pros": ["State-of-the-art accuracy", "Built-in regularization", "Handles missing values"],
                 "cons": ["Many hyperparameters", "Can overfit without tuning", "Slower than LightGBM"]},
                {"rank": 2, "name": "LightGBM Classifier", "suitability_score": 89,
                 "reasoning": "Fast gradient boosting with leaf-wise growth. Great balance of speed and accuracy.",
                 "pros": ["Very fast training", "Low memory usage", "Handles categorical features natively"],
                 "cons": ["Can overfit on small data", "Leaf-wise growth may be noisier", "Fewer community resources than XGBoost"]},
                {"rank": 3, "name": "Random Forest Classifier", "suitability_score": 82,
                 "reasoning": "Reliable ensemble method that provides feature importance.",
                 "pros": ["Robust to overfitting", "Parallelizable", "No feature scaling needed"],
                 "cons": ["Slower than boosting methods", "Large model size", "Less accurate on tabular data than boosting"]},
            ]
        else:
            models = [
                {"rank": 1, "name": "LightGBM Classifier", "suitability_score": 93,
                 "reasoning": f"Optimized for large datasets ({n_rows:,} rows). Histogram-based splitting is fast.",
                 "pros": ["Fastest boosting library", "Handles large data efficiently", "Native categorical support"],
                 "cons": ["Sensitive to hyperparameters", "Leaf-wise can overfit", "Requires careful validation"]},
                {"rank": 2, "name": "XGBoost Classifier", "suitability_score": 88,
                 "reasoning": "Gold standard for tabular classification with extensive tuning options.",
                 "pros": ["Excellent accuracy", "Built-in regularization", "GPU support"],
                 "cons": ["Slower than LightGBM on large data", "Memory intensive", "Complex tuning"]},
                {"rank": 3, "name": "Neural Network (MLP)", "suitability_score": 75,
                 "reasoning": "Deep learning approach that can capture complex non-linear patterns.",
                 "pros": ["Captures complex patterns", "Flexible architecture", "Can leverage GPU"],
                 "cons": ["Requires more data", "Harder to interpret", "Expensive to train"]},
            ]
    else:
        task = "regression"
        if n_rows < 5000:
            models = [
                {"rank": 1, "name": "Ridge Regression", "suitability_score": 80,
                 "reasoning": f"Good baseline for {n_rows:,} rows with L2 regularization to prevent overfitting.",
                 "pros": ["Fast", "Interpretable", "Handles multicollinearity"],
                 "cons": ["Linear only", "Requires scaling", "Cannot capture non-linear relationships"]},
                {"rank": 2, "name": "Random Forest Regressor", "suitability_score": 82,
                 "reasoning": "Captures non-linear patterns and provides feature importance.",
                 "pros": ["Non-linear", "Feature importance", "Robust"],
                 "cons": ["Can overfit small data", "Slower inference", "Less interpretable"]},
                {"rank": 3, "name": "Lasso Regression", "suitability_score": 75,
                 "reasoning": "L1 regularization performs automatic feature selection.",
                 "pros": ["Feature selection built-in", "Fast", "Interpretable"],
                 "cons": ["Linear only", "Can zero out useful features", "Sensitive to scaling"]},
            ]
        else:
            models = [
                {"rank": 1, "name": "XGBoost Regressor", "suitability_score": 90,
                 "reasoning": f"Top performer on tabular regression with {n_rows:,} rows.",
                 "pros": ["State-of-the-art accuracy", "Handles missing data", "Feature importance"],
                 "cons": ["Complex tuning", "Can overfit", "Slower than LightGBM"]},
                {"rank": 2, "name": "LightGBM Regressor", "suitability_score": 88,
                 "reasoning": "Fast gradient boosting optimized for large datasets.",
                 "pros": ["Very fast", "Low memory", "Native categorical support"],
                 "cons": ["Sensitive to hyperparameters", "Leaf-wise can overfit", "Less stable with small data"]},
                {"rank": 3, "name": "Random Forest Regressor", "suitability_score": 80,
                 "reasoning": "Reliable ensemble with built-in feature importance.",
                 "pros": ["Robust", "Parallelizable", "No scaling needed"],
                 "cons": ["Slower", "Large model", "Less accurate than boosting"]},
            ]

    return {
        "task_type": task,
        "target_variable": target,
        "target_reasoning": f"Column '{target}' identified as target ({'categorical' if is_classification else 'numeric'}, {n_unique} unique values).",
        "models": models,
        "preprocessing_steps": [
            f"Handle missing values ({df.isna().sum().sum():,} total missing cells)",
            "Encode categorical variables (One-Hot or Target Encoding)",
            "Scale numeric features (StandardScaler)",
            "Split data into train/test (80/20 stratified)" if is_classification else "Split data into train/test (80/20)",
        ],
        "warnings": [],
    }


def fallback_insights(
    df: pd.DataFrame,
    roles: dict,
    outlier_info: dict,
    corr_info: dict,
) -> list:
    """Generate basic insights from data profile without LLM."""
    insights: List[dict] = []
    rows, cols = df.shape

    # 1. Dataset size
    if rows > 50_000:
        insights.append({
            "title": "Large-Scale Dataset",
            "detail": f"With {rows:,} rows and {cols} columns, this dataset has significant volume. "
                      f"Consider sampling or distributed frameworks for model training.",
            "impact": "medium", "category": "pattern",
            "recommendation": "Use incremental learning or batch processing. LightGBM with histogram binning is optimal.",
        })

    # 2. Missing data
    missing_pct = df.isna().mean().mean() * 100
    if missing_pct > 3:
        worst_col = df.isna().mean().idxmax()
        worst_pct = df[worst_col].isna().mean() * 100
        insights.append({
            "title": "Significant Missing Data Detected",
            "detail": f"Overall {missing_pct:.1f}% of data is missing. "
                      f"Column '{worst_col}' has {worst_pct:.1f}% missing values.",
            "impact": "high" if missing_pct > 15 else "medium",
            "category": "risk",
            "recommendation": f"Investigate why '{worst_col}' has high missing rates. "
                               "Use KNN imputation for numeric and mode imputation for categorical.",
        })

    # 3. Outliers
    if outlier_info.get("total_outliers", 0) > 0:
        top_cols = outlier_info.get("columns", [])[:2]
        col_names = ", ".join(c["name"] for c in top_cols)
        insights.append({
            "title": f"Outliers Detected ({outlier_info['pct_affected_rows']:.1f}% of Rows)",
            "detail": f"{outlier_info['total_outliers']:,} outlier points across "
                      f"{outlier_info['n_columns_affected']} columns. Most affected: {col_names}.",
            "impact": "high" if outlier_info["pct_affected_rows"] > 10 else "medium",
            "category": "anomaly",
            "recommendation": "Review outliers for data quality issues. "
                               "Consider Winsorization (capping at 1st/99th percentile) or robust scaling.",
        })

    # 4. Strong correlations
    strong = corr_info.get("strong_correlations", [])
    if strong:
        top = strong[0]
        insights.append({
            "title": f"Strong Correlation: {top['feature_1']} & {top['feature_2']}",
            "detail": f"Found {len(strong)} strongly correlated pairs (|r| > 0.7). "
                      f"Strongest: r = {top['correlation']:.3f}.",
            "impact": "medium", "category": "pattern",
            "recommendation": "Remove highly correlated features to reduce multicollinearity. "
                               "Use VIF analysis or PCA for dimensionality reduction.",
        })

    # 5. Class imbalance
    for col in roles["categorical"]:
        if df[col].nunique() <= 10:
            vc = df[col].value_counts(normalize=True)
            if vc.min() < 0.1:
                insights.append({
                    "title": f"Class Imbalance in '{col}'",
                    "detail": f"'{col}' has {df[col].nunique()} classes. "
                              f"Minority: '{vc.idxmin()}' at {vc.min() * 100:.1f}%.",
                    "impact": "high", "category": "risk",
                    "recommendation": "Use SMOTE, class weights, or stratified sampling to handle imbalance.",
                })
                break

    # 6. Skewness
    for col in roles["numeric"][:5]:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        skew = float(s.skew())
        if abs(skew) > 2:
            insights.append({
                "title": f"Highly Skewed Distribution: {col}",
                "detail": f"'{col}' has skewness = {skew:.2f} "
                          f"({'right' if skew > 0 else 'left'}-skewed). "
                          f"Range: {s.min():.2f} – {s.max():.2f}.",
                "impact": "low", "category": "pattern",
                "recommendation": f"Apply log or Box-Cox transformation to '{col}' before modeling.",
            })
            break

    if not insights:
        insights.append({
            "title": "Clean Dataset",
            "detail": f"The dataset has {rows:,} rows and {cols} columns with no major issues detected.",
            "impact": "low", "category": "pattern",
            "recommendation": "Proceed directly to feature engineering and model training.",
        })

    return insights[:6]

# ═══════════════════════════════════════════════════════════════════
#   8.  OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_outliers(df: pd.DataFrame, roles: dict) -> dict:
    numeric_cols = roles["numeric"]
    if not numeric_cols:
        return {"total_outliers": 0, "pct_affected_rows": 0.0, "n_columns_affected": 0, "columns": []}

    columns_info: List[dict] = []
    all_outlier_idx: set = set()

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (s < lower) | (s > upper)

        std_val = float(s.std())
        if std_val > 0:
            z = np.abs((s - s.mean()) / std_val)
            z_mask = z > 3
        else:
            z_mask = pd.Series(False, index=s.index)

        combined = iqr_mask | z_mask
        n_out = int(combined.sum())
        if n_out > 0:
            all_outlier_idx.update(s[combined].index.tolist())
            columns_info.append({
                "name": col,
                "n_outliers": n_out,
                "pct": round(n_out / len(s) * 100, 2),
                "iqr_count": int(iqr_mask.sum()),
                "z_count": int(z_mask.sum()),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "min_val": round(float(s.min()), 4),
                "max_val": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
            })

    columns_info.sort(key=lambda x: x["n_outliers"], reverse=True)
    total = len(all_outlier_idx)
    pct = round(total / len(df) * 100, 2) if len(df) > 0 else 0.0

    return {
        "total_outliers": total,
        "pct_affected_rows": pct,
        "n_columns_affected": len(columns_info),
        "columns": columns_info,
    }

# ═══════════════════════════════════════════════════════════════════
#   9.  CORRELATION & FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_correlations(df: pd.DataFrame, roles: dict) -> dict:
    numeric_cols = roles["numeric"]
    if len(numeric_cols) < 2:
        return {"matrix": None, "matrix_columns": [], "top_pairs": [],
                "strong_correlations": [], "n_features": len(numeric_cols), "n_analyzed": 0}

    # pick top 20 by variance to keep heatmap readable
    variances = df[numeric_cols].var(numeric_only=True).dropna().sort_values(ascending=False)
    cols_to_use = variances.head(20).index.tolist()
    if len(cols_to_use) < 2:
        cols_to_use = numeric_cols[:20]

    corr = df[cols_to_use].corr(method="pearson")

    pairs: List[dict] = []
    n = len(cols_to_use)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr.iloc[i, j]
            if np.isnan(val):
                continue
            pairs.append({
                "feature_1": cols_to_use[i],
                "feature_2": cols_to_use[j],
                "correlation": round(float(val), 4),
                "abs_correlation": round(abs(float(val)), 4),
                "direction": "positive" if val > 0 else "negative",
            })

    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    strong = [p for p in pairs if p["abs_correlation"] > 0.7]

    return {
        "matrix": corr,
        "matrix_columns": cols_to_use,
        "top_pairs": pairs[:15],
        "strong_correlations": strong,
        "n_features": len(numeric_cols),
        "n_analyzed": len(cols_to_use),
    }

# ═══════════════════════════════════════════════════════════════════
#  10.  VALIDATION LAYER
# ═══════════════════════════════════════════════════════════════════

class SkipChart(Exception):
    pass


def validate_column(df: pd.DataFrame, col: str, *, required: bool = True) -> None:
    if col is None:
        raise SkipChart("Column not specified")
    if col not in df.columns:
        raise SkipChart(f"Column '{col}' not found")
    if required:
        null_pct = df[col].isna().mean() * 100
        if null_pct > MAX_NULL_PCT_REQUIRED:
            raise SkipChart(f"Column '{col}' is {null_pct:.0f}% null")


def validate_categorical_cardinality(df: pd.DataFrame, col: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
        nuniq = df[col].nunique(dropna=True)
        if nuniq > MAX_UNIQUE_CAT_X:
            raise SkipChart(f"'{col}' has {nuniq:,} unique values — too many")

# ═══════════════════════════════════════════════════════════════════
#  11.  AGGREGATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def _top_n_filter(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    if col not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    top_vals = df[col].value_counts(dropna=True).head(n).index
    return df[df[col].isin(top_vals)]


def aggregate_bar(df, x, y, color, agg="avg", top_n=TOP_N_CAT):
    validate_column(df, x)
    validate_categorical_cardinality(df, x)
    agg = (agg or "count").lower()

    safe_color = color if (color and color in df.columns and color != x) else None
    cols = list(dict.fromkeys([x] + ([y] if y and y in df.columns else []) + ([safe_color] if safe_color else [])))
    d = df[cols].dropna(subset=[x]).copy()
    if d.empty:
        raise SkipChart("All x-values null")

    d = _top_n_filter(d, x, top_n)
    if safe_color:
        d = _top_n_filter(d, safe_color, min(top_n, 12))

    group = [x] + ([safe_color] if safe_color else [])
    if y and y in d.columns and pd.api.types.is_numeric_dtype(d[y]):
        func = {"sum": "sum", "avg": "mean", "mean": "mean", "count": "count"}.get(agg, "mean")
        out = d.groupby(group, dropna=True)[y].agg(func).reset_index()
        y_out = y
    else:
        out = d.groupby(group, dropna=True).size().reset_index(name="count")
        y_out = "count"

    if out.empty:
        raise SkipChart("Empty aggregation")
    if not safe_color:
        out = out.sort_values(y_out, ascending=False).head(top_n)
    return out, y_out, safe_color


def aggregate_line(df, x, y, color, agg="avg"):
    validate_column(df, x)
    validate_column(df, y)
    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        raise SkipChart(f"x='{x}' is not datetime")
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise SkipChart(f"y='{y}' is not numeric")

    safe_color = color if (color and color in df.columns and color != x and color != y) else None
    cols = [x, y] + ([safe_color] if safe_color else [])
    d = df[cols].dropna(subset=[x, y]).copy().sort_values(x)
    if d.empty:
        raise SkipChart("No valid rows")

    span = max((d[x].max() - d[x].min()).days, 1)
    freq = "MS" if span > 365 * 3 else "W" if span > 60 else "D" if span > 7 else "h"
    func = {"sum": "sum", "count": "count"}.get((agg or "avg").lower(), "mean")
    d = d.set_index(x)

    if safe_color:
        top_g = d[safe_color].value_counts(dropna=True).head(8).index
        d = d[d[safe_color].isin(top_g)]
        parts = []
        for key, g in d.groupby(safe_color):
            r = g.resample(freq)[y].agg(func).reset_index()
            r[safe_color] = key
            parts.append(r)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        out = d.resample(freq)[y].agg(func).reset_index()

    if out.empty:
        raise SkipChart("Empty resampled result")
    if len(out) > MAX_POINTS_LINE:
        out = out.tail(MAX_POINTS_LINE)
    return out, safe_color


def prepare_histogram(df, x, color, top_n=TOP_N_CAT):
    validate_column(df, x)
    cols = [x] + ([color] if (color and color in df.columns and color != x) else [])
    d = df[cols].dropna(subset=[x]).copy()
    if d.empty:
        raise SkipChart("No data for histogram")
    if not pd.api.types.is_numeric_dtype(d[x]):
        validate_categorical_cardinality(df, x)
        d = _top_n_filter(d, x, top_n)
    elif len(d) > 200_000:
        d = d.sample(200_000, random_state=42)
    return d


def prepare_scatter(df, x, y, color):
    validate_column(df, x)
    validate_column(df, y)
    cols = list(dict.fromkeys([x, y] + ([color] if (color and color in df.columns) else [])))
    d = df[cols].dropna(subset=[x, y]).copy()
    if d.empty:
        raise SkipChart("No data for scatter")
    if len(d) > MAX_POINTS_SCATTER:
        d = d.sample(MAX_POINTS_SCATTER, random_state=42)
    return d


def prepare_box(df, x, y, color, top_n=TOP_N_CAT):
    validate_column(df, x)
    validate_column(df, y)
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise SkipChart(f"y='{y}' not numeric for box")
    cols = list(dict.fromkeys([x, y] + ([color] if (color and color in df.columns and color != x) else [])))
    d = df[cols].dropna(subset=[x, y]).copy()
    if d.empty:
        raise SkipChart("No data for box")
    d = _top_n_filter(d, x, top_n)
    if color and color in d.columns:
        d = _top_n_filter(d, color, min(top_n, 12))
    if len(d) > 200_000:
        d = d.sample(200_000, random_state=42)
    return d

# ═══════════════════════════════════════════════════════════════════
#  12.  CHART RENDERER
# ═══════════════════════════════════════════════════════════════════

def _apply_theme(fig: go.Figure, title: str, theme_id: str = "midnight") -> go.Figure:
    t = THEMES.get(theme_id, THEMES["midnight"])
    fig.update_layout(
        template=t["plotly_template"],
        title=dict(text=title, font=dict(size=15, color=t["text_primary"]), x=0.01, xanchor="left"),
        font=dict(family=t["font_family"], size=13, color=t["text_secondary"]),
        paper_bgcolor=t["chart_paper_bg"],
        plot_bgcolor=t["chart_bg"],
        margin=dict(l=52, r=24, t=56, b=48),
        colorway=t["chart_colorway"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=t["chart_grid"], zeroline=False,
                     title_font=dict(size=12, color=t["text_secondary"]))
    fig.update_yaxes(showgrid=True, gridcolor=t["chart_grid"], zeroline=False,
                     title_font=dict(size=12, color=t["text_secondary"]))
    return fig


def _skipped_figure(reason: str, theme_id: str = "midnight") -> go.Figure:
    t = THEMES.get(theme_id, THEMES["midnight"])
    fig = go.Figure()
    fig.add_annotation(text=f"⚠ {reason}", xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color=t["text_muted"]))
    fig.update_layout(paper_bgcolor=t["chart_paper_bg"], plot_bgcolor=t["chart_bg"],
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=10), height=220)
    return fig


def build_chart(df: pd.DataFrame, spec: dict, theme_id: str = "midnight") -> go.Figure:
    kind = (spec.get("kind") or "bar").lower()
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    agg = spec.get("agg", "avg")
    top_n = int(spec.get("top_n", TOP_N_CAT))
    title = spec.get("title") or kind.title()

    # sanitize "None" strings
    for name in ("x", "y", "color"):
        val = locals().get(name)
        if val and str(val).lower() == "none":
            if name == "x":   x = None      # noqa
            if name == "y":   y = None      # noqa
            if name == "color": color = None  # noqa

    if color and color == x:
        color = None
    if y and y == x and kind not in ("histogram",):
        y = None
    if color and color not in df.columns:
        color = None

    try:
        if kind == "histogram":
            d = prepare_histogram(df, x, color, top_n)
            sc = color if (color and color in d.columns and color != x) else None
            if not pd.api.types.is_numeric_dtype(d[x]):
                counts = d[x].value_counts().head(top_n).reset_index()
                counts.columns = [x, "count"]
                fig = px.bar(counts, x=x, y="count")
            else:
                fig = px.histogram(d, x=x, color=sc, nbins=50, opacity=0.85)
            return _apply_theme(fig, title, theme_id)

        if kind == "bar":
            out, y_col, sc = aggregate_bar(df, x, y, color, agg, top_n)
            fig = px.bar(out, x=x, y=y_col, color=sc, text_auto=".2s")
            fig.update_traces(textposition="outside", cliponaxis=False)
            return _apply_theme(fig, title, theme_id)

        if kind == "line":
            out, sc = aggregate_line(df, x, y, color, agg)
            fig = px.line(out, x=x, y=y, color=sc, markers=len(out) < 60)
            return _apply_theme(fig, title, theme_id)

        if kind == "scatter":
            d = prepare_scatter(df, x, y, color)
            sc = color if (color and color in d.columns) else None
            fig = px.scatter(d, x=x, y=y, color=sc, opacity=0.6)
            return _apply_theme(fig, title, theme_id)

        if kind == "box":
            d = prepare_box(df, x, y, color, top_n)
            sc = color if (color and color in d.columns and color != x) else None
            fig = px.box(d, x=x, y=y, color=sc)
            return _apply_theme(fig, title, theme_id)

        if kind == "pie":
            validate_column(df, x)
            vc = df[x].value_counts().head(top_n)
            fig = px.pie(values=vc.values, names=vc.index, hole=0.4)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=11)
            return _apply_theme(fig, title, theme_id)

        raise SkipChart(f"Unsupported chart type: {kind}")

    except SkipChart as e:
        log.warning("SKIP chart '%s': %s", title, e)
        return _skipped_figure(str(e), theme_id)
    except Exception as e:
        log.error("ERROR chart '%s': %s", title, e)
        return _skipped_figure(str(e)[:120], theme_id)


# ── Specialized charts ──────────────────────────────────────────

def build_outlier_charts(df: pd.DataFrame, outlier_info: dict, theme_id: str) -> List[str]:
    t = THEMES.get(theme_id, THEMES["midnight"])
    charts: List[str] = []

    top_cols = outlier_info.get("columns", [])[:4]
    if not top_cols:
        return charts

    for ci in top_cols:
        col = ci["name"]
        s = df[col].dropna()
        if s.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=s, name=col,
            marker_color=t["accent_primary"],
            boxpoints="outliers",
            jitter=0.3, pointpos=-1.8,
            marker=dict(outliercolor=t["accent_danger"], size=4, opacity=0.7),
            line=dict(color=t["accent_primary"]),
        ))
        # reference lines
        fig.add_hline(y=ci["upper_bound"], line_dash="dash", line_color=t["accent_warning"],
                      annotation_text=f"Upper: {ci['upper_bound']:.1f}",
                      annotation_font=dict(size=10, color=t["text_muted"]))
        fig.add_hline(y=ci["lower_bound"], line_dash="dash", line_color=t["accent_warning"],
                      annotation_text=f"Lower: {ci['lower_bound']:.1f}",
                      annotation_font=dict(size=10, color=t["text_muted"]))
        fig.add_hline(y=ci["mean"], line_dash="dot", line_color=t["accent_success"],
                      annotation_text=f"Mean: {ci['mean']:.1f}",
                      annotation_font=dict(size=10, color=t["text_muted"]))

        title = f"{col} — {ci['n_outliers']} outliers ({ci['pct']:.1f}%)"
        _apply_theme(fig, title, theme_id)
        fig.update_layout(height=360, showlegend=False)
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return charts


def build_correlation_charts(corr_info: dict, theme_id: str) -> List[str]:
    t = THEMES.get(theme_id, THEMES["midnight"])
    charts: List[str] = []

    matrix = corr_info.get("matrix")
    if matrix is not None and len(matrix) >= 2:
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=[str(c)[:20] for c in matrix.columns],
            y=[str(c)[:20] for c in matrix.index],
            colorscale=t["heatmap_colorscale"],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 9},
            hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(title=dict(text="r", font=dict(size=11, color=t["text_secondary"])),
                          tickfont=dict(size=10, color=t["text_secondary"])),
        ))
        h = max(400, len(matrix) * 28)
        _apply_theme(fig, "Pearson Correlation Matrix", theme_id)
        fig.update_layout(height=h, xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                          yaxis=dict(tickfont=dict(size=10)))
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Top pairs bar
    top_pairs = corr_info.get("top_pairs", [])[:12]
    if top_pairs:
        labels = [f"{p['feature_1'][:15]} × {p['feature_2'][:15]}" for p in top_pairs]
        vals = [p["correlation"] for p in top_pairs]
        colors = [t["accent_primary"] if v > 0 else t["accent_danger"] for v in vals]

        fig = go.Figure(go.Bar(
            y=labels[::-1], x=vals[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in vals[::-1]],
            textposition="outside",
            textfont=dict(size=10),
        ))
        _apply_theme(fig, "Top Correlated Feature Pairs", theme_id)
        fig.update_layout(height=max(350, len(top_pairs) * 32))
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return charts

# ═══════════════════════════════════════════════════════════════════
#  13.  KPI  &  DATA QUALITY
# ═══════════════════════════════════════════════════════════════════

def _fmt_number(v: float) -> str:
    a = abs(v)
    if a >= 1e9:   return f"{v / 1e9:,.2f}B"
    if a >= 1e6:   return f"{v / 1e6:,.2f}M"
    if a >= 1e3:   return f"{v:,.1f}"
    return f"{v:,.2f}"


def compute_kpi(df: pd.DataFrame, kpi: dict) -> str:
    t = kpi.get("type", "")
    col = kpi.get("column")
    try:
        if t == "row_count":
            return f"{len(df):,}"
        if col and col not in df.columns:
            return "n/a"
        s = df[col] if col else None
        if t == "missing_pct" and s is not None:
            return f"{s.isna().mean() * 100:.1f}%"
        if t == "unique_count" and s is not None:
            return f"{s.nunique(dropna=True):,}"
        if t in ("sum", "avg", "max", "min") and s is not None and pd.api.types.is_numeric_dtype(s):
            funcs = {"sum": np.nansum, "avg": np.nanmean, "max": np.nanmax, "min": np.nanmin}
            return _fmt_number(float(funcs[t](s)))
    except Exception:
        pass
    return "n/a"


def compute_quality(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    total = max(rows * cols, 1)
    missing = int(df.isna().sum().sum())
    return {
        "rows": rows, "cols": cols,
        "missing_cells": missing,
        "missing_pct": round(missing / total * 100, 2),
        "dupes": int(df.duplicated().sum()) if rows else 0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 1),
    }

# ═══════════════════════════════════════════════════════════════════
#  14.  TEMPLATE SELECTOR
# ═══════════════════════════════════════════════════════════════════

def select_template(domain_info: dict, override: Optional[str] = None) -> str:
    if override and override in THEMES:
        return override
    domain = (domain_info.get("domain") or "general").lower().strip()
    return DOMAIN_THEME_MAP.get(domain, "midnight")

# ═══════════════════════════════════════════════════════════════════
#  15.  CSS GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_theme_css(theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    return f"""
/* ── Reset & Base ────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:15px;-webkit-font-smoothing:antialiased;scroll-behavior:smooth}}
body{{
  font-family:{t['font_family']};
  background:{t['body_bg']};color:{t['text_primary']};
  line-height:1.6;
}}

/* ── Page Layout ─────────────────────────────────── */
.page{{max-width:1340px;margin:0 auto;padding:36px 32px 72px}}

/* ── Header ──────────────────────────────────────── */
.header{{
  border-bottom:1px solid {t['card_border']};
  padding-bottom:24px;margin-bottom:32px;
}}
.header h1{{
  font-size:1.85rem;font-weight:800;
  letter-spacing:-.03em;
  background:linear-gradient(135deg,{t['gradient_start']},{t['gradient_end']});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
  margin-bottom:6px;
}}
.header .subtitle{{color:{t['text_secondary']};font-size:.85rem}}
.header .theme-badge{{
  display:inline-block;margin-top:8px;padding:3px 12px;border-radius:20px;
  font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  background:{t['kpi_icon_bg']};color:{t['accent_primary']};
}}

/* ── Section Headers ─────────────────────────────── */
.section{{margin-top:44px}}
.section-header{{
  display:flex;align-items:center;gap:14px;
  margin-bottom:20px;
}}
.section-icon{{
  width:40px;height:40px;border-radius:11px;
  display:flex;align-items:center;justify-content:center;
  font-size:20px;flex-shrink:0;
}}
.section-header h2{{font-size:1.2rem;font-weight:700;color:{t['text_primary']}}}
.section-subtitle{{font-size:.82rem;color:{t['text_secondary']};margin-top:2px}}

/* ── KPI Row ─────────────────────────────────────── */
.kpi-row{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
  gap:14px;
}}
.kpi-card{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:14px;
  padding:20px 22px;display:flex;flex-direction:column;gap:8px;
  transition:all .2s ease;box-shadow:{t['shadow']};
}}
.kpi-card:hover{{
  border-color:{t['card_border_hover']};
  box-shadow:{t['shadow_hover']};
  transform:translateY(-2px);
}}
.kpi-label{{
  font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:{t['text_secondary']};
}}
.kpi-value{{font-size:1.6rem;font-weight:800;color:{t['text_primary']}}}

/* ── Chart Grid ──────────────────────────────────── */
.chart-grid-2{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}
.chart-grid-1{{display:grid;grid-template-columns:1fr;gap:16px}}
@media(max-width:900px){{.chart-grid-2{{grid-template-columns:1fr}}}}
.chart-card{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:16px;
  padding:16px 18px 10px;overflow:hidden;
  box-shadow:{t['shadow']};transition:all .2s ease;
}}
.chart-card:hover{{
  border-color:{t['card_border_hover']};
  box-shadow:{t['shadow_hover']};
  transform:translateY(-1px);
}}

/* ── Cards ───────────────────────────────────────── */
.card{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:14px;
  padding:24px;box-shadow:{t['shadow']};transition:all .2s ease;
}}
.card:hover{{border-color:{t['card_border_hover']};box-shadow:{t['shadow_hover']}}}
.card h3{{font-size:1rem;font-weight:700;margin-bottom:16px;color:{t['text_primary']}}}

/* ── Outlier Summary ─────────────────────────────── */
.summary-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px}}
.summary-card{{
  background:{t['card_bg_alt']};border:1px solid {t['card_border']};border-radius:12px;
  padding:18px 20px;text-align:center;
}}
.summary-card .sv{{font-size:1.8rem;font-weight:800;color:{t['accent_warning']};display:block}}
.summary-card .sl{{font-size:.72rem;color:{t['text_secondary']};text-transform:uppercase;letter-spacing:.05em;margin-top:4px}}

/* ── Outlier Table ───────────────────────────────── */
.outlier-table{{overflow-x:auto;margin-top:16px}}
.outlier-table table{{width:100%;border-collapse:collapse;font-size:.85rem}}
.outlier-table th{{
  text-align:left;padding:10px 14px;border-bottom:2px solid {t['card_border']};
  font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
  color:{t['text_secondary']};font-weight:600;
}}
.outlier-table td{{
  padding:10px 14px;border-bottom:1px solid {t['card_border']};
  color:{t['text_primary']};
}}
.outlier-table tr:hover td{{background:{t['card_bg_alt']}}}

/* ── Correlation Pairs ───────────────────────────── */
.corr-pair{{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 16px;border-bottom:1px solid {t['card_border']};font-size:.88rem;
}}
.corr-pair:last-child{{border-bottom:none}}
.corr-pair .cp-names{{color:{t['text_primary']}}}
.corr-pair .cp-val{{font-weight:700;font-family:monospace}}
.cp-pos{{color:{t['accent_primary']}}}
.cp-neg{{color:{t['accent_danger']}}}

/* ── ML Recommendation ───────────────────────────── */
.task-badge{{
  display:inline-block;padding:6px 18px;border-radius:24px;
  font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  background:linear-gradient(135deg,{t['gradient_start']},{t['gradient_end']});
  color:#fff;margin-bottom:12px;
}}
.ml-reason{{color:{t['text_secondary']};font-size:.9rem;margin-bottom:20px;line-height:1.6}}
.target-box{{
  background:{t['card_bg_alt']};border-left:4px solid {t['accent_primary']};
  padding:14px 20px;border-radius:0 10px 10px 0;margin-bottom:20px;
  font-size:.9rem;color:{t['text_primary']};
}}
.target-box strong{{color:{t['accent_primary']}}}
.model-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}}
.model-card{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:14px;
  padding:22px;position:relative;overflow:hidden;
  box-shadow:{t['shadow']};transition:all .2s ease;
}}
.model-card:hover{{border-color:{t['card_border_hover']};box-shadow:{t['shadow_hover']};transform:translateY(-2px)}}
.model-card .rank{{
  position:absolute;top:0;right:0;
  background:linear-gradient(135deg,{t['gradient_start']},{t['gradient_end']});
  color:#fff;padding:6px 14px;border-radius:0 14px 0 14px;
  font-size:.75rem;font-weight:800;
}}
.model-card h4{{font-size:1.05rem;font-weight:700;margin-bottom:6px;color:{t['text_primary']};padding-right:50px}}
.score-bar{{
  height:8px;border-radius:4px;background:{t['card_bg_alt']};margin:12px 0 10px;overflow:hidden;
}}
.score-fill{{height:100%;border-radius:4px;background:linear-gradient(90deg,{t['gradient_start']},{t['gradient_end']})}}
.score-label{{font-size:.75rem;color:{t['text_secondary']};margin-bottom:10px}}
.model-card .reason{{font-size:.84rem;color:{t['text_secondary']};line-height:1.55;margin-bottom:12px}}
.pros-cons{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.pros-cons ul{{padding-left:16px;font-size:.78rem;line-height:1.6}}
.pros-cons .pros-title{{color:{t['accent_success']};font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}}
.pros-cons .cons-title{{color:{t['accent_danger']};font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}}
.pros-cons li{{color:{t['text_secondary']}}}

/* ── Preprocessing Steps ─────────────────────────── */
.prep-steps{{padding-left:20px;color:{t['text_secondary']};font-size:.88rem;line-height:1.8}}
.prep-steps li::marker{{color:{t['accent_primary']}}}

/* ── Insights ────────────────────────────────────── */
.insights-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}
.insight-card{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:14px;
  padding:22px 24px;box-shadow:{t['shadow']};transition:all .2s ease;
  border-top:3px solid transparent;
}}
.insight-card:hover{{border-color:{t['card_border_hover']};box-shadow:{t['shadow_hover']};transform:translateY(-2px)}}
.insight-card.impact-high{{border-top-color:{t['accent_danger']}}}
.insight-card.impact-medium{{border-top-color:{t['accent_warning']}}}
.insight-card.impact-low{{border-top-color:{t['accent_success']}}}
.insight-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.insight-badge{{
  padding:3px 10px;border-radius:12px;font-size:.68rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.04em;
}}
.badge-trend{{background:#3b82f620;color:#60a5fa}}
.badge-risk{{background:#ef444420;color:#f87171}}
.badge-opportunity{{background:#22c55e20;color:#4ade80}}
.badge-pattern{{background:#8b5cf620;color:#a78bfa}}
.badge-anomaly{{background:#f9731620;color:#fb923c}}
.badge-recommendation{{background:#06b6d420;color:#22d3ee}}
.insight-impact{{font-size:.7rem;font-weight:600;text-transform:uppercase;color:{t['text_muted']}}}
.insight-card h4{{font-size:.95rem;font-weight:700;margin-bottom:8px;color:{t['text_primary']}}}
.insight-card .detail{{font-size:.85rem;color:{t['text_secondary']};line-height:1.6;margin-bottom:12px}}
.insight-rec{{
  background:{t['card_bg_alt']};border-radius:8px;padding:10px 14px;
  font-size:.82rem;color:{t['text_primary']};line-height:1.5;
}}
.insight-rec strong{{color:{t['accent_primary']};font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}}

/* ── Info Panels ─────────────────────────────────── */
.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}}
@media(max-width:760px){{.info-grid{{grid-template-columns:1fr}}}}
.quality-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;
}}
.quality-item{{display:flex;flex-direction:column;gap:3px}}
.quality-item .ql{{font-size:.7rem;color:{t['text_secondary']};text-transform:uppercase;letter-spacing:.04em}}
.quality-item .qv{{font-size:1.08rem;font-weight:700;color:{t['text_primary']}}}
.nar-list{{padding-left:18px;color:{t['text_secondary']};font-size:.88rem;line-height:1.75}}
.nar-list li::marker{{color:{t['text_muted']}}}

/* ── Footer ──────────────────────────────────────── */
.footer{{
  margin-top:56px;padding-top:20px;border-top:1px solid {t['card_border']};
  display:flex;justify-content:space-between;align-items:center;
  color:{t['text_muted']};font-size:.75rem;
}}
.print-btn{{
  background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:8px;
  padding:6px 16px;color:{t['text_secondary']};font-size:.78rem;cursor:pointer;
  transition:all .15s;
}}
.print-btn:hover{{border-color:{t['accent_primary']};color:{t['accent_primary']}}}

/* ── Print Styles ────────────────────────────────── */
@media print{{
  body{{background:#fff!important;color:#000!important}}
  .card,.chart-card,.kpi-card,.model-card,.insight-card,.summary-card{{
    background:#fff!important;border:1px solid #ddd!important;color:#000!important;
    box-shadow:none!important;
  }}
  .no-print{{display:none!important}}
  .header h1{{-webkit-text-fill-color:#000;color:#000}}
}}
"""

# ═══════════════════════════════════════════════════════════════════
#  16.  HTML SECTION RENDERERS
# ═══════════════════════════════════════════════════════════════════

_esc = html_lib.escape


def _render_header(title: str, quality: dict, theme_id: str, domain_info: dict) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    domain_label = (domain_info.get("domain", "general")).replace("_", " ").title()
    sub_domain = domain_info.get("sub_domain", "")
    return f"""
  <div class="header">
    <h1>{_esc(title)}</h1>
    <div class="subtitle">
      {quality['rows']:,} rows &middot; {quality['cols']} columns &middot;
      {quality['memory_mb']:.1f} MB
    </div>
    <span class="theme-badge">{_esc(domain_label)}{(' — ' + _esc(sub_domain)) if sub_domain and sub_domain != domain_label.lower() else ''} &middot; {_esc(t['name'])} Theme</span>
  </div>"""


def _render_kpis(kpis: List[dict]) -> str:
    cards = ""
    for k in kpis:
        cards += f"""
        <div class="kpi-card">
          <span class="kpi-label">{_esc(str(k['label']))}</span>
          <span class="kpi-value">{_esc(str(k['value']))}</span>
        </div>"""
    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{THEMES.get('midnight',{}).get('kpi_icon_bg','#6366f120')}">📊</div>
      <div><h2>Key Metrics</h2></div>
    </div>
    <div class="kpi-row">{cards}
    </div>
  </div>"""


def _render_charts_section(charts_html: List[str], label: str, icon: str) -> str:
    if not charts_html:
        return ""
    primary = charts_html[:2]
    secondary = charts_html[2:]
    p_block = "\n".join(f'<div class="chart-card">{c}</div>' for c in primary)
    s_block = "\n".join(f'<div class="chart-card">{c}</div>' for c in secondary)

    out = f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#3b82f620">{icon}</div>
      <div><h2>{_esc(label)}</h2><div class="section-subtitle">Interactive charts — hover, zoom, and click to explore</div></div>
    </div>
    <div class="chart-grid-2">{p_block}</div>"""
    if s_block:
        out += f'\n    <div class="chart-grid-2" style="margin-top:16px">{s_block}</div>'
    out += "\n  </div>"
    return out


def _render_outlier_section(outlier_info: dict, outlier_charts: List[str], theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    if outlier_info.get("total_outliers", 0) == 0 and not outlier_charts:
        return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_warning']}22">⚠️</div>
      <div><h2>Outlier Analysis</h2><div class="section-subtitle">IQR and Z-score methods</div></div>
    </div>
    <div class="card"><p style="color:{t['text_secondary']}">No significant outliers detected in numeric columns.</p></div>
  </div>"""

    # summary cards
    summary = f"""
    <div class="summary-row">
      <div class="summary-card">
        <span class="sv">{outlier_info['total_outliers']:,}</span>
        <span class="sl">Total Outlier Points</span>
      </div>
      <div class="summary-card">
        <span class="sv">{outlier_info['pct_affected_rows']:.1f}%</span>
        <span class="sl">Rows Affected</span>
      </div>
      <div class="summary-card">
        <span class="sv">{outlier_info['n_columns_affected']}</span>
        <span class="sl">Columns with Outliers</span>
      </div>
    </div>"""

    # charts
    chart_block = ""
    if outlier_charts:
        chart_block = '<div class="chart-grid-2">' + \
            "\n".join(f'<div class="chart-card">{c}</div>' for c in outlier_charts) + \
            "</div>"

    # detail table
    rows_html = ""
    for ci in outlier_info.get("columns", [])[:10]:
        rows_html += f"""<tr>
          <td><strong>{_esc(ci['name'])}</strong></td>
          <td>{ci['n_outliers']:,}</td>
          <td>{ci['pct']:.1f}%</td>
          <td style="font-family:monospace;font-size:.8rem">[{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]</td>
          <td>IQR: {ci['iqr_count']:,} &nbsp;|&nbsp; Z: {ci['z_count']:,}</td>
        </tr>"""

    table = f"""
    <div class="card outlier-table" style="margin-top:16px">
      <h3>Outlier Details by Column</h3>
      <table>
        <tr><th>Column</th><th>Count</th><th>%</th><th>IQR Bounds</th><th>Detection Method</th></tr>
        {rows_html}
      </table>
    </div>""" if rows_html else ""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_warning']}22">⚠️</div>
      <div><h2>Outlier Analysis</h2><div class="section-subtitle">Statistical outlier detection using IQR (1.5×) and Z-score (&gt;3σ) methods</div></div>
    </div>
    {summary}
    {chart_block}
    {table}
  </div>"""


def _render_correlation_section(corr_info: dict, corr_charts: List[str], theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    n_features = corr_info.get("n_features", 0)
    if n_features < 2:
        return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_secondary']}22">🔗</div>
      <div><h2>Correlation &amp; Feature Analysis</h2><div class="section-subtitle">Feature relationships</div></div>
    </div>
    <div class="card"><p style="color:{t['text_secondary']}">Need at least 2 numeric columns for correlation analysis.</p></div>
  </div>"""

    chart_block = ""
    if corr_charts:
        if len(corr_charts) == 1:
            chart_block = '<div class="chart-grid-1">' + f'<div class="chart-card">{corr_charts[0]}</div>' + "</div>"
        else:
            chart_block = '<div class="chart-grid-2">' + \
                "\n".join(f'<div class="chart-card">{c}</div>' for c in corr_charts) + \
                "</div>"

    # strong correlations list
    strong = corr_info.get("strong_correlations", [])
    pairs_html = ""
    if strong:
        for p in strong[:8]:
            cls = "cp-pos" if p["direction"] == "positive" else "cp-neg"
            sign = "+" if p["correlation"] > 0 else ""
            pairs_html += f"""
            <div class="corr-pair">
              <span class="cp-names">{_esc(p['feature_1'])} &harr; {_esc(p['feature_2'])}</span>
              <span class="cp-val {cls}">{sign}{p['correlation']:.3f}</span>
            </div>"""
        pairs_html = f"""
        <div class="card" style="margin-top:16px">
          <h3>Strongly Correlated Pairs (|r| &gt; 0.7)</h3>
          {pairs_html}
        </div>"""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_secondary']}22">🔗</div>
      <div>
        <h2>Correlation &amp; Feature Analysis</h2>
        <div class="section-subtitle">{corr_info.get('n_analyzed', 0)} of {n_features} numeric features analyzed (top by variance)</div>
      </div>
    </div>
    {chart_block}
    {pairs_html}
  </div>"""


def _render_ml_section(ml_info: dict, theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    task = ml_info.get("task_type", "unknown").replace("_", " ").title()
    target = ml_info.get("target_variable")
    reasoning = ml_info.get("target_reasoning", "")
    models = ml_info.get("models", [])
    steps = ml_info.get("preprocessing_steps", [])
    warnings = ml_info.get("warnings", [])

    # target box
    target_html = ""
    if target:
        target_html = f"""
    <div class="target-box">
      <strong>Target Variable:</strong> {_esc(str(target))}
      {"<br><span style='font-size:.82rem;color:" + t['text_secondary'] + "'>" + _esc(reasoning) + "</span>" if reasoning else ""}
    </div>"""
    elif reasoning:
        target_html = f'<div class="target-box">{_esc(reasoning)}</div>'

    # model cards
    model_cards = ""
    for m in models[:3]:
        rank = m.get("rank", "")
        score = m.get("suitability_score", 0)
        pros_li = "".join(f"<li>{_esc(str(p))}</li>" for p in m.get("pros", []))
        cons_li = "".join(f"<li>{_esc(str(c))}</li>" for c in m.get("cons", []))
        model_cards += f"""
        <div class="model-card">
          <span class="rank">#{rank}</span>
          <h4>{_esc(m.get('name', 'Model'))}</h4>
          <div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>
          <div class="score-label">Suitability: {score}/100</div>
          <div class="reason">{_esc(m.get('reasoning', ''))}</div>
          <div class="pros-cons">
            <div><div class="pros-title">✓ Pros</div><ul>{pros_li}</ul></div>
            <div><div class="cons-title">✗ Cons</div><ul>{cons_li}</ul></div>
          </div>
        </div>"""

    # preprocessing
    steps_html = ""
    if steps:
        steps_li = "".join(f"<li>{_esc(str(s))}</li>" for s in steps)
        steps_html = f"""
    <div class="card" style="margin-top:16px">
      <h3>Recommended Preprocessing Pipeline</h3>
      <ol class="prep-steps">{steps_li}</ol>
    </div>"""

    # warnings
    warn_html = ""
    if warnings:
        warn_items = "".join(f"<li style='color:{t['accent_warning']}'>{_esc(str(w))}</li>" for w in warnings)
        warn_html = f'<ul style="margin-top:12px;padding-left:18px;font-size:.85rem">{warn_items}</ul>'

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_primary']}22">🤖</div>
      <div>
        <h2>Machine Learning Recommendation</h2>
        <div class="section-subtitle">Model suggestions based on dataset characteristics, outliers, and correlations</div>
      </div>
    </div>
    <div class="task-badge">{_esc(task)}</div>
    {target_html}
    <div class="model-grid">{model_cards}</div>
    {steps_html}
    {warn_html}
  </div>"""


def _render_insights_section(insights: list, domain_info: dict, theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    if not insights:
        return ""
    domain_label = (domain_info.get("domain", "general")).replace("_", " ").title()

    cards = ""
    for ins in insights:
        impact = (ins.get("impact") or "medium").lower()
        cat = (ins.get("category") or "pattern").lower()
        badge_cls = f"badge-{cat}" if cat in ("trend", "risk", "opportunity", "pattern", "anomaly", "recommendation") else "badge-pattern"
        cards += f"""
        <div class="insight-card impact-{impact}">
          <div class="insight-header">
            <span class="insight-badge {badge_cls}">{_esc(cat)}</span>
            <span class="insight-impact">{_esc(impact)} impact</span>
          </div>
          <h4>{_esc(ins.get('title', 'Insight'))}</h4>
          <div class="detail">{_esc(ins.get('detail', ''))}</div>
          <div class="insight-rec">
            <strong>Recommendation</strong><br>
            {_esc(ins.get('recommendation', 'No specific recommendation.'))}
          </div>
        </div>"""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:{t['accent_success']}22">💡</div>
      <div>
        <h2>{_esc(domain_label)} Insights</h2>
        <div class="section-subtitle">{'AI-generated' if GROQ_API_KEY else 'Auto-generated'} production-grade analysis</div>
      </div>
    </div>
    <div class="insights-grid">{cards}</div>
  </div>"""


def _render_quality_section(quality: dict, narrative: list, theme_id: str) -> str:
    t = THEMES.get(theme_id, THEMES["midnight"])
    nar_items = "".join(f"<li>{_esc(str(n))}</li>" for n in (narrative or []))
    q = quality
    return f"""
  <div class="section">
    <div class="info-grid">
      <div class="card">
        <h3>Data Quality</h3>
        <div class="quality-grid">
          <div class="quality-item"><span class="ql">Rows</span><span class="qv">{q['rows']:,}</span></div>
          <div class="quality-item"><span class="ql">Columns</span><span class="qv">{q['cols']}</span></div>
          <div class="quality-item"><span class="ql">Duplicates</span><span class="qv">{q['dupes']:,}</span></div>
          <div class="quality-item"><span class="ql">Missing</span><span class="qv">{q['missing_cells']:,} ({q['missing_pct']:.1f}%)</span></div>
          <div class="quality-item"><span class="ql">Memory</span><span class="qv">{q['memory_mb']:.1f} MB</span></div>
        </div>
      </div>
      <div class="card">
        <h3>Analyst Summary</h3>
        <ul class="nar-list">{nar_items if nar_items else '<li>No additional notes.</li>'}</ul>
      </div>
    </div>
  </div>"""


def _render_footer() -> str:
    return """
  <div class="footer">
    <span>Generated by Pro Dashboard Generator v2</span>
    <button class="print-btn no-print" onclick="window.print()">🖨 Export PDF</button>
  </div>"""

# ═══════════════════════════════════════════════════════════════════
#  17.  HTML ASSEMBLER
# ═══════════════════════════════════════════════════════════════════

def render_html(
    *,
    title: str,
    theme_id: str,
    domain_info: dict,
    kpis: List[dict],
    charts_html: List[str],
    outlier_info: dict,
    outlier_charts: List[str],
    corr_info: dict,
    corr_charts: List[str],
    ml_info: dict,
    insights: list,
    quality: dict,
    narrative: list,
) -> str:
    css = generate_theme_css(theme_id)
    safe_title = _esc(title)

    sections = [
        _render_header(title, quality, theme_id, domain_info),
        _render_kpis(kpis),
        _render_charts_section(charts_html, "Primary Analysis", "📈"),
        _render_outlier_section(outlier_info, outlier_charts, theme_id),
        _render_correlation_section(corr_info, corr_charts, theme_id),
        _render_ml_section(ml_info, theme_id),
        _render_insights_section(insights, domain_info, theme_id),
        _render_quality_section(quality, narrative, theme_id),
        _render_footer(),
    ]

    body = "\n".join(s for s in sections if s)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{safe_title}</title>
<script src="{PLOTLY_CDN}"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════
#  18.  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def _generate_html_filename(input_path: str) -> str:
    stem = Path(input_path).stem
    return str(OUTPUTS_DIR / f"{stem}_dashboard_v2.html")


def _ensure_io_dirs() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _persist_input_file(input_path: str) -> str:
    src = Path(input_path).resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    ext = src.suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"Unsupported file type: {ext}")
    if src.parent == INPUTS_DIR.resolve():
        return str(src)
    dest = INPUTS_DIR / src.name
    if dest.exists():
        dest = INPUTS_DIR / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}{src.suffix}"
    shutil.copy2(src, dest)
    log.info("Saved input copy → %s", dest)
    return str(dest)


def _summarize_for_llm(info: dict, max_chars: int = 1500) -> str:
    """Compact string summary for LLM context."""
    text = json.dumps(info, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def main(
    input_path: str,
    out_html: Optional[str] = None,
    sheet: Optional[str] = None,
    use_groq: bool = False,
    theme_override: Optional[str] = None,
) -> str:
    """
    Generate a production-grade HTML dashboard.
    Returns the path to the generated HTML file.
    """
    _ensure_io_dirs()
    input_path = _persist_input_file(input_path)

    if out_html is None:
        out_html = _generate_html_filename(input_path)
    else:
        out_html = str(OUTPUTS_DIR / Path(out_html).name)

    # ── 1. Load & process ──────────────────────────────────────
    df = load_table(input_path, sheet=sheet)
    df = process_dataframe(df)
    roles = get_column_roles(df)
    summary = dataframe_summary(df)
    quality = compute_quality(df)

    log.info("Roles: %d numeric, %d categorical, %d datetime",
             len(roles["numeric"]), len(roles["categorical"]), len(roles["datetime"]))

    # ── 2. Analytics ───────────────────────────────────────────
    outlier_info = detect_outliers(df, roles)
    log.info("Outliers: %d points in %d columns (%.1f%% rows)",
             outlier_info["total_outliers"], outlier_info["n_columns_affected"],
             outlier_info["pct_affected_rows"])

    corr_info = compute_correlations(df, roles)
    log.info("Correlations: %d pairs analyzed, %d strong (|r|>0.7)",
             len(corr_info.get("top_pairs", [])), len(corr_info.get("strong_correlations", [])))

    # ── 3. LLM or fallback ─────────────────────────────────────
    client = _get_groq_client() if use_groq else None

    # 3a. Domain detection
    if client:
        domain_info = detect_domain_llm(summary, client)
        time.sleep(LLM_CALL_DELAY)
    else:
        domain_info = detect_domain_rules(df)
    log.info("Domain: %s (%s)", domain_info.get("domain"), domain_info.get("sub_domain"))

    # 3b. Theme selection
    theme_id = select_template(domain_info, theme_override)
    log.info("Theme: %s (%s)", theme_id, THEMES[theme_id]["name"])

    # 3c. Chart specification
    if client:
        try:
            spec = ask_groq_for_spec(summary, client)
            log.info("LLM spec: %d charts, %d KPIs", len(spec.get("charts", [])), len(spec.get("kpis", [])))
            time.sleep(LLM_CALL_DELAY)
        except Exception as e:
            log.warning("Groq spec failed (%s) → using default", e)
            spec = default_spec(df)
    else:
        spec = default_spec(df)

    # 3d. ML recommendation
    outlier_str = _summarize_for_llm(outlier_info)
    corr_str = _summarize_for_llm({
        "n_features": corr_info["n_features"],
        "strong_correlations": corr_info.get("strong_correlations", [])[:5],
        "top_pairs": corr_info.get("top_pairs", [])[:8],
    })

    if client:
        try:
            ml_info = ask_groq_for_ml(summary, outlier_str, corr_str, client)
            log.info("ML recommendation: %s → %s", ml_info.get("task_type"),
                     ", ".join(m.get("name", "") for m in ml_info.get("models", [])[:3]))
            time.sleep(LLM_CALL_DELAY)
        except Exception as e:
            log.warning("ML recommendation LLM failed (%s) → fallback", e)
            ml_info = fallback_ml_recommendation(df, roles)
    else:
        ml_info = fallback_ml_recommendation(df, roles)

    # 3e. Domain insights
    ml_str = _summarize_for_llm(ml_info)
    if client:
        try:
            insights = ask_groq_for_insights(
                summary, domain_info.get("domain", "general"),
                outlier_str, corr_str, ml_str, client,
            )
            log.info("AI insights: %d generated", len(insights))
        except Exception as e:
            log.warning("Insights LLM failed (%s) → fallback", e)
            insights = fallback_insights(df, roles, outlier_info, corr_info)
    else:
        insights = fallback_insights(df, roles, outlier_info, corr_info)

    # ── 4. KPIs ────────────────────────────────────────────────
    kpis = []
    for k in (spec.get("kpis") or [])[:MAX_KPIS]:
        kpis.append({"label": k.get("label", "KPI"), "value": compute_kpi(df, k)})
    if not kpis:
        kpis = [{"label": "Rows", "value": f"{len(df):,}"}]

    # ── 5. Charts ──────────────────────────────────────────────
    chart_specs = (spec.get("charts") or [])[:MAX_CHARTS]
    if not chart_specs:
        chart_specs = default_spec(df).get("charts", [])

    charts_html: List[str] = []
    for cs in chart_specs:
        fig = build_chart(df, cs, theme_id)
        charts_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ── 6. Specialized charts ──────────────────────────────────
    outlier_charts = build_outlier_charts(df, outlier_info, theme_id)
    corr_charts = build_correlation_charts(corr_info, theme_id)

    # ── 7. Assemble HTML ───────────────────────────────────────
    html_str = render_html(
        title=str(spec.get("title", "Data Analysis Dashboard")),
        theme_id=theme_id,
        domain_info=domain_info,
        kpis=kpis,
        charts_html=charts_html,
        outlier_info=outlier_info,
        outlier_charts=outlier_charts,
        corr_info=corr_info,
        corr_charts=corr_charts,
        ml_info=ml_info,
        insights=insights,
        quality=quality,
        narrative=spec.get("narrative") or [],
    )

    Path(out_html).write_text(html_str, encoding="utf-8")
    log.info("Dashboard saved → %s", out_html)
    return out_html

# ═══════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pro Dashboard Generator v2")
    parser.add_argument("input", nargs="?", default=None, help="Path to CSV / XLSX / XLS file")
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument("--sheet", default=None, help="Excel sheet name")
    parser.add_argument("--groq", action="store_true", help="Use Groq LLM for analysis")
    parser.add_argument("--theme", default=None, choices=list(THEMES.keys()),
                        help="Override theme (default: auto-detect from domain)")
    args = parser.parse_args()

    input_file = args.input
    if not input_file:
        input_file = input("\n📊 Enter path to CSV/XLSX file: ").strip()
        if not input_file:
            print("❌ No file provided. Exiting.")
            exit(1)

    result = main(
        input_file,
        out_html=args.out,
        sheet=args.sheet,
        use_groq=args.groq,
        theme_override=args.theme,
    )
    print(f"\n✅ Dashboard: {result}")
