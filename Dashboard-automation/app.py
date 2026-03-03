from pathlib import Path

import streamlit as st

from dashboard import INPUTS_DIR, OUTPUTS_DIR, _ensure_io_dirs, main

st.set_page_config(page_title="Dashboard Generator", page_icon="📊", layout="centered")

st.title("📊 Auto Dashboard Generator")
st.caption("Upload CSV/XLSX, generate a meaningful interactive dashboard, and save files in organized folders.")

_ensure_io_dirs()

uploaded = st.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls"])
use_groq = st.checkbox("Use Groq for AI chart specification", value=False)
output_name = st.text_input("Output file name (optional)", value="")

if st.button("Generate Dashboard", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Please upload a dataset first.")
    else:
        input_path = INPUTS_DIR / uploaded.name
        if input_path.exists():
            stem = input_path.stem
            suffix = input_path.suffix
            counter = 1
            while input_path.exists():
                input_path = INPUTS_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        input_path.write_bytes(uploaded.getbuffer())

        final_output_name = output_name.strip() or None
        if final_output_name and not final_output_name.lower().endswith(".html"):
            final_output_name += ".html"

        with st.spinner("Generating dashboard..."):
            main(
                str(input_path),
                out_html=final_output_name,
                use_groq=use_groq,
            )

        if final_output_name:
            output_path = OUTPUTS_DIR / Path(final_output_name).name
        else:
            output_path = OUTPUTS_DIR / f"{input_path.stem}_dashboard.html"

        st.success(f"Dashboard generated: {output_path.name}")
        st.write(f"Input saved in: {INPUTS_DIR}")
        st.write(f"Output saved in: {output_path}")

        with output_path.open("rb") as f:
            st.download_button(
                "Download Dashboard HTML",
                data=f,
                file_name=output_path.name,
                mime="text/html",
                use_container_width=True,
            )
