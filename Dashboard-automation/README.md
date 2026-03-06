# Auto Dashboard Generator

**Professional data visualization made simple.** Upload any CSV/XLSX dataset and instantly generate interactive HTML dashboards with charts, KPIs, and insights—no coding required.

Built with Python, Plotly, Streamlit, and optional LLM-powered chart recommendations.

## Features

✅ **Automatic data type detection** — coerces numeric strings and detects datetime columns
✅ **Smart chart generation** — histograms, bar charts, line trends, scatter plots, box plots
✅ **LLM integration** — optionally use Groq to pick optimal chart specs
✅ **Validation layer** — skips invalid/empty charts with clear logging
✅ **Professional design** — dark GitHub-style theme, responsive layout
✅ **Large dataset support** — handles 500k+ rows efficiently
✅ **Auto-naming** — HTML files named after dataset (e.g., `banking_dataset_dashboard.html`)
✅ **Simple web UI** — upload data and generate dashboard using Streamlit
✅ **Organized storage** — uploaded files saved in `inputs/`, dashboards saved in `outputs/`

## Setup

### 1. Requirements

```bash
pip install pandas plotly groq python-dotenv openpyxl numpy streamlit
```

### 2. Environment Configuration

Create a `.env` file in the script directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Project Structure

```
Dashboard-automation/
├── app.py                 # Streamlit web UI for file upload & dashboard generation
├── dashboard.py           # Core dashboard generation engine (CLI & programmatic)
├── .env                   # Environment configuration (API keys)
├── README.md             # This file
├── inputs/               # Uploaded/processed datasets (auto-created)
├── outputs/              # Generated HTML dashboards (auto-created)
├── datasets/             # Sample datasets for testing
├── archive/              # Previous versions (app_v2.py, dashboard_v2.py)
└── __pycache__/          # Python bytecode cache
```

### Architecture Overview

**Frontend:** `app.py` — Streamlit-based web interface

- File upload (CSV/XLSX/XLS)
- Dashboard generation controls
- Download generated HTML

**Backend:** `dashboard.py` — Data processing & visualization engine

- Automatic data type coercion
- Column role detection (numeric, datetime, categorical)
- Smart chart selection (default or LLM-powered)
- Plotly chart rendering with consistent theming
- HTML dashboard assembly

**Storage:**

- Input files → `inputs/` (persisted with timestamp if duplicate names)
- Output dashboards → `outputs/` (named after dataset or custom name)

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install pandas plotly groq python-dotenv openpyxl numpy streamlit
   ```

2. **Launch the web UI:**

   ```bash
   streamlit run app.py
   ```

3. **Upload & Generate:**
   - Drag & drop your CSV/XLSX file
   - Click **Generate Dashboard**
   - Download the interactive HTML report

That's it! Your dashboard is ready to share.

## Usage

### Front-end (recommended)

```bash
streamlit run app.py
```

Then:

- Upload a `.csv`, `.xlsx`, or `.xls` file
- Click **Generate Dashboard**
- Input file is stored in `inputs/`
- Dashboard HTML is stored in `outputs/`

### Basic (Auto-generated filename)

```bash
python dashboard.py banking_dataset.csv
# → Saves a copy in inputs/
# → Creates: outputs/banking_dataset_dashboard.html
```

### With custom output name

```bash
python dashboard.py banking_dataset.csv --out my_report.html
# → Creates: outputs/my_report.html
```

### Interactive prompt

```bash
python dashboard.py
# 📊 Enter path to CSV/XLSX file: banking_dataset.csv
# → Saves input in inputs/ and dashboard in outputs/
```

### With LLM spec generation

```bash
python dashboard.py data.xlsx --groq
# Uses Groq to analyze data and recommend charts
```

### Excel sheet selection

```bash
python dashboard.py workbook.xlsx --sheet "Sales Data"
```

## Examples

```bash
# Quick analysis of CSV
python dashboard.py sales.csv

# Large dataset with LLM
python dashboard.py customer_data.xlsx --groq

# Custom name + specific sheet
python dashboard.py financial_data.xlsx --sheet "Q4 2025" --out q4_analysis.html
```

## Output

Generates a single-page HTML dashboard with:

- **Header** — dataset name, row/column counts
- **KPI Row** — key metrics (totals, averages, unique counts)
- **Primary Charts** (2-column grid) — main analysis visualizations
- **Secondary Charts** (2-column grid) — additional insights
- **Data Quality Panel** — missing values, duplicates, memory usage
- **Analyst Summary** — auto-generated insights

Storage locations:

- `inputs/` → saved dataset files
- `outputs/` → generated dashboard HTML files

All dashboards include Plotly.js (CDN-loaded) for interactive exploration.

## How It Works

The dashboard generation follows a robust 7-stage pipeline:

1. **Load & Coerce** — Read CSV/XLSX and auto-convert numeric/datetime strings
2. **Profile** — Analyze column types (numeric, categorical, datetime) and distributions
3. **Spec** — Generate chart specifications (LLM-powered or rule-based)
4. **Validate** — Pre-check columns for nulls, cardinality, data quality issues
5. **Aggregate** — Prepare data: group-by, time-resample, top-N filtering
6. **Render** — Build Plotly figures with consistent dark theme
7. **Assemble** — Generate single-page HTML with embedded charts and metadata

Invalid charts are gracefully skipped with detailed logging. The system prioritizes reliability over completeness.

## Notes

- **File Organization:** All uploaded datasets are automatically saved in `inputs/` and dashboards in `outputs/` for easy version control
- **Large Datasets:** Files with 200k+ rows are intelligently sampled for scatter/histogram rendering (aggregations use full data)
- **Security:** API keys are read from `.env` file — never hardcoded or committed to git
- **Error Handling:** Invalid chart specs are logged and skipped gracefully; partial failures don't crash the entire dashboard
- **Date Detection:** Datetime columns auto-detected from date-like strings (requires >60% match rate)
- **Numeric Coercion:** Columns with commas, currency symbols (₹$€£¥), and percentages are auto-converted to numeric types
- **Archive:** Previous versions of scripts are stored in `archive/` folder

## Troubleshooting

**Dashboard not generating?**

- Check that uploaded file is CSV/XLSX/XLS format
- Verify file isn't corrupted (try opening in Excel first)
- Check console logs for specific error messages

**Charts missing or skipped?**

- Column may have >80% null values (threshold configurable)
- Categorical columns with >100k unique values are skipped for bar charts
- Review console output for "SKIP" or "ERROR" messages

**LLM errors?**

- Verify `GROQ_API_KEY` is set in `.env` file
- Check your Groq API quota/rate limits
- Falls back to rule-based charts if LLM fails

## License

Open source. Use freely.
