"""
Pro Dashboard Generator v2 — Streamlit App
============================================
Enhanced UI for generating production-grade dashboards with AI insights.
"""

from pathlib import Path

import streamlit as st

from dashboard_v2 import (
    INPUTS_DIR,
    OUTPUTS_DIR,
    THEMES,
    _ensure_io_dirs,
    main,
)

# ── Page config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Auto Data Intelligence Engine v3",
    page_icon="📊",
    layout="centered",
)

_ensure_io_dirs()

# ── Header ───────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center;padding:16px 0 8px">
        <h1 style="margin-bottom:4px">📊 Auto Data Intelligence Engine <sup style="font-size:.55em;color:#818cf8">v3</sup></h1>
        <p style="color:#8b949e;font-size:.9rem;margin:0">
            Upload ANY dataset → get a full analytics dashboard, auto-trained ML model,
            evaluation charts, feature importance, segmentation, anomaly detection,
            and an in-browser prediction simulator.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── File upload ──────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX, XLS",
)

# ── Options ──────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    use_groq = st.toggle(
        "Use AI for analysis",
        value=True,
        help="Uses Groq LLM (Llama 3.3 70B) for chart specs, ML recommendations, domain detection, and rich insights. "
             "Falls back to rule-based analysis if disabled or if API calls fail.",
    )

with col2:
    theme_options = {"auto": "🎨 Auto-detect (recommended)"}
    for tid, tdata in THEMES.items():
        theme_options[tid] = f"{tdata['name']}"

    theme_choice = st.selectbox(
        "Dashboard Theme",
        options=list(theme_options.keys()),
        format_func=lambda k: theme_options[k],
        help="Auto-detect picks a theme based on the dataset domain (finance → Ocean, medical → Clinical, etc.)",
    )

col3, col4 = st.columns(2)

with col3:
    enable_ml = st.toggle(
        "Auto ML Pipeline",
        value=True,
        help="Enables v3 features: auto target detection, model training, evaluation charts, "
             "feature importance, segmentation, anomaly detection, and in-browser prediction.",
    )

with col4:
    st.caption("")
    st.caption(
        "v3 adds: 🤖 Auto-trained model · 🎯 Feature importance · "
        "🧩 Segmentation · 🔍 Anomaly detection · 🧪 Live predictor"
    )

output_name = st.text_input(
    "Output file name (optional)",
    value="",
    placeholder="e.g., my_analysis.html — leave empty for auto-naming",
)

# ── Generate ─────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Generate Dashboard", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Please upload a dataset first.")
    else:
        # Save uploaded file
        input_path = INPUTS_DIR / uploaded.name
        if input_path.exists():
            stem, suffix = input_path.stem, input_path.suffix
            counter = 1
            while input_path.exists():
                input_path = INPUTS_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        input_path.write_bytes(uploaded.getbuffer())

        final_output_name = output_name.strip() or None
        if final_output_name and not final_output_name.lower().endswith(".html"):
            final_output_name += ".html"

        theme_override = theme_choice if theme_choice != "auto" else None

        # Progress display
        progress_bar = st.progress(0, text="Starting analysis...")

        try:
            progress_bar.progress(10, text="Loading and processing data...")

            result_path = main(
                str(input_path),
                out_html=final_output_name,
                use_groq=use_groq,
                theme_override=theme_override,
                disable_ml=not enable_ml,
            )

            progress_bar.progress(100, text="Dashboard generated!")

            output_path = Path(result_path)

            # Success message
            st.success(f"Dashboard generated: **{output_path.name}**")

            # Info columns
            ic1, ic2 = st.columns(2)
            with ic1:
                st.caption(f"📂 Input saved: `{INPUTS_DIR}`")
            with ic2:
                st.caption(f"📂 Output saved: `{output_path.parent}`")

            # Download button
            with output_path.open("rb") as f:
                st.download_button(
                    "⬇️ Download Dashboard HTML",
                    data=f,
                    file_name=output_path.name,
                    mime="text/html",
                    use_container_width=True,
                )

            # Preview hint
            st.info(
                "💡 Open the downloaded HTML file in any browser for full interactivity. "
                "Charts support hover, zoom, and pan.",
                icon="ℹ️",
            )

        except Exception as e:
            progress_bar.empty()
            st.error(f"Dashboard generation failed: {e}")
            st.exception(e)

# ── Footer ───────────────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#484f58;font-size:.75rem;padding:8px 0">
        <strong>Auto Data Intelligence Engine v3</strong> &middot;
        6 themes &middot; AI insights &middot; Outlier &amp; Anomaly detection &middot;
        Correlation heatmaps &middot; Auto ML &middot; Segmentation &middot; Live Predictor<br>
        Powered by Plotly + scikit-learn + ONNX Runtime + Groq (Llama 3.3 70B)
    </div>
    """,
    unsafe_allow_html=True,
)
