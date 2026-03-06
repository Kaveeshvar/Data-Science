"""
ml_engine.html_sections
========================
HTML section renderers for v3 ML features.

Each function returns an HTML string that can be injected into the dashboard.
Maintains dark-theme consistency with the existing v2 sections.
"""

from __future__ import annotations

import html as html_lib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_esc = html_lib.escape


# ═══════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE SECTION
# ═══════════════════════════════════════════════════════════════════

def render_model_performance_section(
    eval_result: Dict[str, Any],
    train_result: Dict[str, Any],
    theme_id: str,
) -> str:
    """Render 🤖 Model Performance section."""
    metrics = eval_result.get("metrics", {})
    charts = eval_result.get("charts_html", [])
    warnings = eval_result.get("warnings", [])
    problem_type = metrics.get("problem_type", train_result.get("problem_type", ""))
    model_type = train_result.get("model_type", "Unknown")
    cv_results = train_result.get("cv_results", [])

    # Metric cards
    metric_cards = ""
    if problem_type == "classification":
        for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"),
                           ("f1_weighted", "F1 (weighted)"), ("pr_auc", "PR AUC")]:
            val = metrics.get(key)
            if val is not None:
                metric_cards += f"""
                <div class="summary-card">
                    <span class="sv">{val:.4f}</span>
                    <span class="sl">{_esc(label)}</span>
                </div>"""
    else:
        for key, label in [("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE")]:
            val = metrics.get(key)
            if val is not None:
                fmt = f"{val:.4f}" if key == "r2" else f"{val:,.2f}"
                metric_cards += f"""
                <div class="summary-card">
                    <span class="sv">{fmt}</span>
                    <span class="sl">{_esc(label)}</span>
                </div>"""

    # CV comparison
    cv_html = ""
    if cv_results:
        rows = ""
        for r in cv_results:
            score = r.get("mean_score")
            if score is not None:
                std = r.get("std_score", 0)
                rows += f"<tr><td><strong>{_esc(r['name'])}</strong></td><td>{score:.4f}</td><td>±{std:.4f}</td></tr>"
        if rows:
            cv_html = f"""
            <div class="card" style="margin-top:16px">
                <h3>Cross-Validation Comparison</h3>
                <div class="outlier-table"><table>
                    <tr><th>Model</th><th>Mean Score</th><th>Std Dev</th></tr>
                    {rows}
                </table></div>
            </div>"""

    # Warning cards
    warn_html = ""
    if warnings:
        items = "".join(
            f'<div style="background:#f8717120;border:1px solid #f8717140;border-radius:10px;'
            f'padding:12px 16px;font-size:.85rem;color:#fca5a5;margin-bottom:8px">'
            f'⚠️ {_esc(w)}</div>' for w in warnings
        )
        warn_html = f'<div style="margin-bottom:16px">{items}</div>'

    # Charts
    chart_block = ""
    if charts:
        chart_block = '<div class="chart-grid-2">' + \
            "\n".join(f'<div class="chart-card">{c}</div>' for c in charts) + \
            "</div>"

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#818cf822">🤖</div>
      <div>
        <h2>Model Performance</h2>
        <div class="section-subtitle">Auto-trained {_esc(model_type)} — evaluated on 20% held-out test set</div>
      </div>
    </div>
    {warn_html}
    <div class="summary-row">{metric_cards}</div>
    {chart_block}
    {cv_html}
  </div>"""


# ═══════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE SECTION
# ═══════════════════════════════════════════════════════════════════

def render_feature_importance_section(
    train_result: Dict[str, Any],
    theme_id: str,
) -> str:
    """Render feature importance bar chart + positive/negative drivers."""
    importance = train_result.get("feature_importance", {})
    signed = train_result.get("signed_importances", {})
    model_type = train_result.get("model_type", "")

    if not importance:
        return ""

    # Top 15 for bar chart
    top15 = dict(list(importance.items())[:15])

    import plotly.graph_objects as go
    names = list(reversed(list(top15.keys())))
    vals = list(reversed(list(top15.values())))

    # Clean names for display
    display_names = [n.replace("num__", "").replace("cat__", "")[:30] for n in names]

    fig = go.Figure(go.Bar(
        y=display_names, x=vals, orientation="h",
        marker_color="#818cf8",
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Top 15 Feature Importances",
        height=max(400, len(top15) * 30),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance" if "Forest" in model_type or "Tree" in model_type else "|Coefficient|",
        margin=dict(l=200),
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    # Positive / Negative drivers (for linear models)
    drivers_html = ""
    if signed:
        sorted_signed = sorted(signed.items(), key=lambda x: x[1], reverse=True)
        positives = [(k.replace("num__", "").replace("cat__", ""), v)
                     for k, v in sorted_signed if v > 0][:5]
        negatives = [(k.replace("num__", "").replace("cat__", ""), v)
                     for k, v in sorted_signed if v < 0][-5:]

        if positives or negatives:
            pos_items = "".join(
                f'<li><strong>{_esc(n)}</strong>: +{v:.4f}</li>' for n, v in positives
            ) if positives else "<li>None detected</li>"
            neg_items = "".join(
                f'<li><strong>{_esc(n)}</strong>: {v:.4f}</li>' for n, v in negatives
            ) if negatives else "<li>None detected</li>"

            drivers_html = f"""
            <div class="info-grid" style="margin-top:16px">
                <div class="card">
                    <h3 style="color:#34d399">📈 Top Positive Drivers</h3>
                    <p style="font-size:.82rem;color:#8b949e;margin-bottom:12px">
                        Features that increase the predicted value / probability
                    </p>
                    <ul class="prep-steps">{pos_items}</ul>
                </div>
                <div class="card">
                    <h3 style="color:#f87171">📉 Top Negative Drivers</h3>
                    <p style="font-size:.82rem;color:#8b949e;margin-bottom:12px">
                        Features that decrease the predicted value / probability
                    </p>
                    <ul class="prep-steps">{neg_items}</ul>
                </div>
            </div>"""
    else:
        # Tree-based: explain differently
        top3 = list(top15.items())[:3]
        if top3:
            items = "".join(
                f'<li><strong>{_esc(n.replace("num__","").replace("cat__",""))}</strong> '
                f'(importance: {v:.4f})</li>'
                for n, v in top3
            )
            drivers_html = f"""
            <div class="card" style="margin-top:16px">
                <h3>🔑 Key Drivers (Tree-Based Feature Importance)</h3>
                <p style="font-size:.82rem;color:#8b949e;margin-bottom:12px">
                    Feature importance is based on mean decrease in impurity across all trees.
                    Higher values indicate features that contribute more to prediction accuracy.
                </p>
                <ul class="prep-steps">{items}</ul>
            </div>"""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#a78bfa22">🎯</div>
      <div>
        <h2>Feature Importance &amp; Model Explanation</h2>
        <div class="section-subtitle">Understanding what drives the {_esc(model_type)} model predictions</div>
      </div>
    </div>
    <div class="chart-grid-1"><div class="chart-card">{chart_html}</div></div>
    {drivers_html}
  </div>"""


# ═══════════════════════════════════════════════════════════════════
#  SEGMENTATION SECTION
# ═══════════════════════════════════════════════════════════════════

def render_segmentation_section(
    seg_result: Dict[str, Any],
    theme_id: str,
) -> str:
    """Render 🧩 Customer Segments section."""
    n_clusters = seg_result["n_clusters"]
    sil = seg_result["silhouette"]
    charts = seg_result.get("charts_html", [])
    explanations = seg_result.get("explanations", [])
    profiles: pd.DataFrame = seg_result.get("profiles")

    # Explanation cards
    expl_html = ""
    if explanations:
        items = "".join(
            f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;'
            f'padding:14px 18px;margin-bottom:10px;font-size:.85rem;color:#e6edf3;line-height:1.6">'
            f'{_esc(e)}</div>' for e in explanations
        )
        expl_html = f'<div style="margin-top:16px">{items}</div>'

    # Profile table
    table_html = ""
    if profiles is not None and not profiles.empty:
        cols = profiles.columns.tolist()[:10]
        header = "<th>Cluster</th>" + "".join(f"<th>{_esc(str(c)[:20])}</th>" for c in cols)
        rows = ""
        for idx, row in profiles.iterrows():
            cells = "".join(f"<td>{row[c]:.2f}</td>" for c in cols)
            rows += f"<tr><td><strong>Cluster {idx}</strong></td>{cells}</tr>"
        table_html = f"""
        <div class="card outlier-table" style="margin-top:16px">
            <h3>Cluster Feature Profiles (Mean Values)</h3>
            <table><tr>{header}</tr>{rows}</table>
        </div>"""

    # Charts
    chart_block = ""
    if charts:
        chart_block = '<div class="chart-grid-2">' + \
            "\n".join(f'<div class="chart-card">{c}</div>' for c in charts) + \
            "</div>"

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#22d3ee22">🧩</div>
      <div>
        <h2>Customer Segments</h2>
        <div class="section-subtitle">KMeans clustering — {n_clusters} segments (silhouette score: {sil:.3f})</div>
      </div>
    </div>
    <div class="summary-row">
      <div class="summary-card">
        <span class="sv">{n_clusters}</span>
        <span class="sl">Optimal Clusters</span>
      </div>
      <div class="summary-card">
        <span class="sv">{sil:.3f}</span>
        <span class="sl">Silhouette Score</span>
      </div>
    </div>
    {chart_block}
    {expl_html}
    {table_html}
  </div>"""


# ═══════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION SECTION
# ═══════════════════════════════════════════════════════════════════

def render_anomaly_section(
    anom_result: Dict[str, Any],
    theme_id: str,
) -> str:
    """Render 🔍 Advanced Anomaly Detection section."""
    n = anom_result["n_anomalies"]
    pct = anom_result["pct_anomalies"]
    charts = anom_result.get("charts_html", [])
    explanation = anom_result.get("explanation", "")
    top_df: pd.DataFrame = anom_result.get("top_anomalies")
    stats = anom_result.get("score_stats", {})

    # Charts
    chart_block = ""
    if charts:
        chart_block = '<div class="chart-grid-2">' + \
            "\n".join(f'<div class="chart-card">{c}</div>' for c in charts) + \
            "</div>"

    # Top anomalies table
    table_html = ""
    if top_df is not None and not top_df.empty:
        cols = top_df.columns.tolist()
        header = "".join(f"<th>{_esc(str(c)[:25])}</th>" for c in cols)
        rows = ""
        for _, row in top_df.head(10).iterrows():
            cells = ""
            for c in cols:
                val = row[c]
                if isinstance(val, float):
                    cells += f"<td>{val:.4f}</td>"
                else:
                    cells += f"<td>{_esc(str(val)[:30])}</td>"
            rows += f"<tr>{cells}</tr>"
        table_html = f"""
        <div class="card outlier-table" style="margin-top:16px">
            <h3>Top 10 Most Anomalous Records</h3>
            <table><tr>{header}</tr>{rows}</table>
        </div>"""

    # Explanation
    expl_html = ""
    if explanation:
        expl_html = f"""
        <div class="card" style="margin-top:16px">
            <h3>Analysis</h3>
            <p style="font-size:.88rem;color:#8b949e;line-height:1.7">{_esc(explanation)}</p>
        </div>"""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#f8717122">🔍</div>
      <div>
        <h2>Advanced Anomaly Detection</h2>
        <div class="section-subtitle">IsolationForest-based anomaly analysis — complements IQR/Z-score outlier detection</div>
      </div>
    </div>
    <div class="summary-row">
      <div class="summary-card">
        <span class="sv">{n:,}</span>
        <span class="sl">Anomalies Detected</span>
      </div>
      <div class="summary-card">
        <span class="sv">{pct:.1f}%</span>
        <span class="sl">Anomaly Rate</span>
      </div>
      <div class="summary-card">
        <span class="sv">{stats.get('min', 0):.3f}</span>
        <span class="sl">Min Anomaly Score</span>
      </div>
    </div>
    {chart_block}
    {expl_html}
    {table_html}
  </div>"""


# ═══════════════════════════════════════════════════════════════════
#  IN-BROWSER PREDICTION SIMULATOR
# ═══════════════════════════════════════════════════════════════════

def render_prediction_simulator(
    inference_meta: Dict[str, Any],
    onnx_available: bool,
    theme_id: str,
    onnx_model_b64: Optional[str] = None,
) -> str:
    """
    Render 🧪 Live Prediction Simulator section.

    If ONNX bytes are provided, embeds them as base64 for in-browser inference.
    Otherwise, renders the form with info about using exported model.
    """
    features = inference_meta.get("features", [])
    problem_type = inference_meta.get("problemType", "")
    target = inference_meta.get("target", "")
    label_map = inference_meta.get("labelMap")
    num_features = inference_meta.get("numericFeatures", [])
    cat_features = inference_meta.get("categoricalFeatures", [])

    if not features:
        return ""

    meta_json = json.dumps(inference_meta, default=str)

    # Build form fields
    form_fields = ""
    for feat in features:
        name = feat["name"]
        safe_name = _esc(name)
        field_id = f"pred_{name.replace(' ', '_').replace('.', '_')}"

        if feat["type"] == "numeric":
            default = feat.get("default", 0)
            mn = feat.get("min", 0)
            mx = feat.get("max", 100)
            form_fields += f"""
            <div class="pred-field">
                <label for="{field_id}">{safe_name}</label>
                <input type="number" id="{field_id}" name="{_esc(name)}"
                       value="{default}" step="any" min="{mn}" max="{mx}"
                       class="pred-input" data-type="numeric">
                <span class="pred-hint">Range: {mn:.2f} – {mx:.2f}</span>
            </div>"""
        else:
            cats = feat.get("categories", [])
            default = feat.get("default", "")
            options = "".join(
                f'<option value="{_esc(str(c))}" {"selected" if str(c) == str(default) else ""}>'
                f'{_esc(str(c))}</option>'
                for c in cats
            )
            form_fields += f"""
            <div class="pred-field">
                <label for="{field_id}">{safe_name}</label>
                <select id="{field_id}" name="{_esc(name)}" class="pred-input" data-type="categorical">
                    {options}
                </select>
            </div>"""

    # ONNX script
    onnx_script = ""
    if onnx_available and onnx_model_b64:
        onnx_script = f"""
        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
        <script>
        (function() {{
            const meta = window.__predMeta;
            const onnxModelB64 = "{onnx_model_b64}";
            let session = null;

            async function loadModel() {{
                try {{
                    // Decode base64 to bytes
                    const bstr = atob(onnxModelB64);
                    const len = bstr.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {{
                        bytes[i] = bstr.charCodeAt(i);
                    }}
                    
                    // Load ONNX model
                    session = await ort.InferenceSession.create(bytes);
                    document.getElementById('pred-status').textContent = '✅ Model loaded in-browser — ready for predictions (no server needed!)';
                    document.getElementById('pred-status').style.color = '#34d399';
                }} catch(e) {{
                    document.getElementById('pred-status').textContent = '❌ Model load failed: ' + e.message;
                    document.getElementById('pred-status').style.color = '#f87171';
                }}
            }}

            window.runPrediction = async function() {{
                if (!session) {{ alert('⏳ Model is still loading...'); return; }}
                try {{
                    const formInputs = {{}};
                    for (const f of meta.features) {{
                        const el = document.querySelector('[name="' + f.name + '"]');
                        if (el) {{
                            formInputs[f.name] = f.type === 'numeric' ? parseFloat(el.value) || 0 : el.value;
                        }}
                    }}
                    
                    // Build feature vector in correct order
                    const numericFeats = meta.numericFeatures || [];
                    const categoricalFeats = meta.categoricalFeatures || [];
                    const allFeats = numericFeats.concat(categoricalFeats);
                    
                    const inputs = [];
                    
                    // Add numeric features
                    for (const fname of numericFeats) {{
                        inputs.push(formInputs[fname] || 0);
                    }}
                    
                    // Add categorical (one-hot encoded)
                    for (const fname of categoricalFeats) {{
                        const feat = meta.features.find(f => f.name === fname);
                        if (feat && feat.categories) {{
                            const val = formInputs[fname] || '';
                            for (const cat of feat.categories) {{
                                inputs.push(cat === val ? 1.0 : 0.0);
                            }}
                        }}
                    }}
                    
                    const tensor = new ort.Tensor('float32', new Float32Array(inputs), [1, inputs.length]);
                    const inputName = session.inputNames[0];
                    const results = await session.run({{ [inputName]: tensor }});
                    const outputName = session.outputNames[0];
                    const output = results[outputName].data;

                    let resultHTML = '';
                    if (meta.problemType === 'classification') {{
                        let predicted = output[0];
                        // For multi-class, take argmax
                        if (output.length > 1) {{
                            predicted = output.indexOf(Math.max(...Array.from(output)));
                        }}
                        
                        const labelMap = meta.labelMap || {{}};
                        const invMap = {{}};
                        for (const [k, v] of Object.entries(labelMap)) {{
                            invMap[v] = k;
                        }}
                        const label = invMap[predicted] || predicted;
                        resultHTML = '<div class="pred-result"><strong>✅ Predicted Class:</strong> ' + label + '</div>';
                        
                        // Confidence
                        if (output.length > 0) {{
                            const maxProb = Math.max(...Array.from(output));
                            resultHTML += '<div class="pred-confidence"><strong>Confidence:</strong> ' +
                                         (maxProb * 100).toFixed(1) + '%</div>';
                            resultHTML += '<div class="confidence-bar"><div class="confidence-fill" style="width:' +
                                         (Math.min(maxProb * 100, 100)) + '%"></div></div>';
                        }}
                    }} else {{
                        resultHTML = '<div class="pred-result"><strong>✅ Predicted Value:</strong> ' +
                                    parseFloat(output[0]).toFixed(4) + '</div>';
                    }}
                    document.getElementById('pred-output').innerHTML = resultHTML;
                }} catch(e) {{
                    document.getElementById('pred-output').innerHTML =
                        '<div style="color:#f87171">❌ Prediction error: ' + e.message + '</div>';
                }}
            }};

            loadModel();
        }})();
        </script>"""
    elif onnx_available:
        # ONNX available but no base64 provided
        onnx_script = """
        <script>
        window.runPrediction = function() {
            document.getElementById('pred-output').innerHTML =
                '<div style="color:#facc15;padding:12px;border-radius:6px;background:#fef3c755">' +
                '⚠️ <strong>Model bytes not embedded</strong><br>' +
                'The ONNX model export may have failed. Use <code>model.pkl</code> with Python for predictions.<br><br>' +
                '<code style="background:#1e1e1e;color:#4ec9b0;padding:6px 10px;border-radius:4px;display:inline-block;margin-top:8px;font-size:0.85rem">' +
                'import pickle<br>with open("model.pkl", "rb") as f:<br>&nbsp;&nbsp;model = pickle.load(f)<br>prediction = model.predict([[...]])</code>' +
                '</div>';
        };
        </script>"""
    else:
        onnx_script = """
        <script>
        window.runPrediction = function() {
            document.getElementById('pred-output').innerHTML =
                '<div style="color:#facc15;padding:12px;border-radius:6px;background:#fef3c755">' +
                '⚠️ <strong>In-browser prediction unavailable</strong><br>' +
                'ONNX model export was skipped or failed (model type may not be supported). ' +
                'Use the exported <code>model.pkl</code> file with Python:<br><br>' +
                '<code style="background:#1e1e1e;color:#4ec9b0;padding:6px 10px;border-radius:4px;display:inline-block;margin-top:8px;font-size:0.85rem">' +
                'import pickle<br>import pandas as pd<br><br>' +
                'model = pickle.load(open("model.pkl", "rb"))<br>' +
                'prediction = model.predict(pd.DataFrame([[...]], columns=[...]))</code>' +
                '</div>';
        };
        </script>"""

    return f"""
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:#34d39922">🧪</div>
      <div>
        <h2>Live Prediction Simulator</h2>
        <div class="section-subtitle">Enter values and predict — {'runs 100% in-browser via embedded ONNX (no server needed)' if (onnx_available and onnx_model_b64) else 'preview form (export model for live predictions)'}</div>
      </div>
    </div>
    <div class="card">
      <p id="pred-status" style="font-size:.82rem;color:#8b949e;margin-bottom:16px">
        {'🔄 Loading model into browser...' if (onnx_available and onnx_model_b64) else '⚠️ Model not embedded for in-browser inference'}
      </p>
      <div class="pred-form">
        {form_fields}
      </div>
      <div style="margin-top:20px">
        <button onclick="runPrediction()" class="pred-btn">🔮 Predict</button>
      </div>
      <div id="pred-output" style="margin-top:16px"></div>
    </div>
  </div>
  <script>
    window.__predMeta = {meta_json};
  </script>
  {onnx_script}"""


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL FILTER PANEL
# ═══════════════════════════════════════════════════════════════════

def render_global_filter_panel(
    df: pd.DataFrame,
    categorical_cols: List[str],
    theme_id: str,
) -> str:
    """
    Render a lightweight JS filter panel for top categorical columns.
    Uses dropdown filters that add ?filter params to URL (no re-render, just UI).
    """
    # Take top 5 categorical columns with reasonable cardinality
    filter_cols = []
    for col in categorical_cols[:5]:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 30:
            top_vals = df[col].value_counts().head(20).index.tolist()
            filter_cols.append({"name": col, "values": [str(v) for v in top_vals]})

    if not filter_cols:
        return ""

    # Build dropdowns
    dropdowns = ""
    for fc in filter_cols:
        options = '<option value="">All</option>' + "".join(
            f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in fc["values"]
        )
        dropdowns += f"""
        <div class="filter-item">
            <label>{_esc(fc['name'])}</label>
            <select class="filter-select" data-column="{_esc(fc['name'])}">{options}</select>
        </div>"""

    return f"""
  <div class="section" id="filter-panel">
    <div class="section-header">
      <div class="section-icon" style="background:#facc1522">🔧</div>
      <div>
        <h2>Filter Panel</h2>
        <div class="section-subtitle">Filter data by category — affects summary statistics display</div>
      </div>
    </div>
    <div class="card">
      <div class="filter-grid">{dropdowns}</div>
      <div id="filter-status" style="margin-top:12px;font-size:.82rem;color:#8b949e"></div>
    </div>
  </div>
  <script>
  (function() {{
    const selects = document.querySelectorAll('.filter-select');
    selects.forEach(sel => {{
      sel.addEventListener('change', () => {{
        const active = [];
        selects.forEach(s => {{
          if (s.value) active.push(s.dataset.column + '=' + s.value);
        }});
        const status = document.getElementById('filter-status');
        if (active.length) {{
          status.textContent = 'Active filters: ' + active.join(', ');
          status.style.color = '#facc15';
        }} else {{
          status.textContent = 'No filters active';
          status.style.color = '#8b949e';
        }}
      }});
    }});
  }})();
  </script>"""


# ═══════════════════════════════════════════════════════════════════
#  ADDITIONAL CSS FOR V3 SECTIONS
# ═══════════════════════════════════════════════════════════════════

def get_v3_css() -> str:
    """Return additional CSS for v3 sections."""
    return """
/* ── v3 Prediction Simulator ─────────────────── */
.pred-form{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;
}
.pred-field{display:flex;flex-direction:column;gap:4px}
.pred-field label{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#8b949e}
.pred-input{
  background:#0d1117;border:1px solid #21262d;border-radius:8px;
  padding:10px 14px;color:#e6edf3;font-size:.9rem;
  transition:border-color .15s;
}
.pred-input:focus{outline:none;border-color:#818cf8}
.pred-hint{font-size:.7rem;color:#484f58}
.pred-btn{
  background:linear-gradient(135deg,#6366f1,#a78bfa);color:#fff;
  border:none;border-radius:10px;padding:12px 32px;font-size:.95rem;
  font-weight:700;cursor:pointer;transition:all .2s;
}
.pred-btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(99,102,241,.4)}
.pred-result{
  background:#161b22;border:1px solid #21262d;border-radius:10px;
  padding:16px 20px;font-size:1.1rem;color:#e6edf3;margin-bottom:8px;
}
.pred-confidence{font-size:.9rem;color:#8b949e;margin-bottom:8px}
.confidence-bar{height:8px;border-radius:4px;background:#1c2128;overflow:hidden;max-width:300px}
.confidence-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#6366f1,#a78bfa)}

/* ── v3 Filter Panel ─────────────────────────── */
.filter-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;
}
.filter-item{display:flex;flex-direction:column;gap:4px}
.filter-item label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#8b949e}
.filter-select{
  background:#0d1117;border:1px solid #21262d;border-radius:8px;
  padding:8px 12px;color:#e6edf3;font-size:.85rem;
}
.filter-select:focus{outline:none;border-color:#818cf8}
"""
