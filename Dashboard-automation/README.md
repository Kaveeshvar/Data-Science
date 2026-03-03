# Auto Dashboard Generator

Generate professional, interactive HTML dashboards from CSV/XLSX files with automatic chart selection, data coercion, optional LLM-powered analysis, and a simple front-end uploader.

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

## Notes

- Large datasets (>200k rows) are intelligently sampled for scatter/histogram rendering
- API key is read from `.env` file — **never hardcoded**
- All invalid chart specs are logged and skipped gracefully
- Datetime columns auto-detected from date-like strings (>60% match)
- Numeric strings with commas/currency symbols are auto-converted

## License

Open source. Use freely.
