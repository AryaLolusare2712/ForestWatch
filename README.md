# ForestWatch

> Satellite-based forest monitoring and vegetation-change analysis.

[![Live dashboard](https://img.shields.io/badge/Live%20dashboard-Vercel-22c55e?style=for-the-badge&logo=vercel&logoColor=white)](https://forest-watch-mu.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-facc15?style=for-the-badge)](LICENSE)

ForestWatch helps users compare dated satellite observations, evaluate vegetation-health indicators, surface potential change, and prepare monitoring insights. It combines a React GIS dashboard with a FastAPI service and a local, dataset-driven analysis workflow.

## Live dashboard

Explore the deployed app: **[forest-watch-mu.vercel.app](https://forest-watch-mu.vercel.app/)**

<p align="center">
  <img src="docs/screenshots/forestwatch-dashboard.png" alt="ForestWatch dashboard showing satellite imagery, change metrics, and a selected-period distribution chart" width="100%">
</p>

## What it does

- Compares earlier and later satellite observations.
- Calculates vegetation-health and NDVI-based change indicators.
- Displays stable, decrease, and increase areas in an interactive dashboard.
- Shows true-colour imagery alongside change summaries.
- Supports alerts, reporting, prediction experiments, and image uploads.
- Keeps preprocessing non-destructive and records a processing manifest.

## How it works

```text
Satellite observations
        │
        ▼
Dataset scanning and preprocessing
        │
        ▼
NDVI and image analysis
        │
        ├──► Change detection
        ├──► Health alerts
        ├──► Historical prediction
        └──► Reports
        │
        ▼
React GIS dashboard + FastAPI API
```

## Tech stack

| Area | Tools |
| --- | --- |
| Dashboard | React, Vite |
| API | FastAPI, Uvicorn |
| Geospatial analysis | Rasterio, NumPy, Pillow |
| Data and persistence | Pandas, SQLAlchemy |
| Visualisation | Plotly, Matplotlib |
| Modelling | Scikit-learn |

## Project structure

```text
ForestWatch/
├── frontend/           # React/Vite GIS dashboard
├── forestwatch/        # FastAPI service and analysis modules
├── tests/              # Test suite
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Local container setup
└── .env.example        # Environment-variable reference
```

## Run locally

### Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- A local satellite dataset (optional for exploring the interface, required for meaningful analysis)

### 1. Clone and configure Python

```bash
git clone https://github.com/AryaLolusare2712/ForestWatch.git
cd ForestWatch
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the API dependencies:

```bash
pip install -r requirements.txt
```

### 2. Create local configuration

Copy `.env.example` to `.env`, then adjust the paths and thresholds for your machine.

```env
FORESTWATCH_DATASET=dataset
FORESTWATCH_OUTPUT=data
FORESTWATCH_PROJECT_DATA=dataset
FORESTWATCH_LOW_NDVI=0.3
FORESTWATCH_HEALTHY_NDVI=0.6
FORESTWATCH_DECREASE=-0.15
```

Never commit `.env` files, SMTP passwords, or private dataset paths.

### 3. Start the API

```bash
python -m uvicorn forestwatch.api:app --reload
```

The API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start the dashboard

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal—normally [http://127.0.0.1:5173](http://127.0.0.1:5173).

### Optional: run the Gradio workspace

```bash
python -m forestwatch.gradio_app
```

## Dataset notes

ForestWatch works with dated Sentinel-1 and Sentinel-2 observations. Organise source material by date so the dashboard can compare periods consistently.

```text
dataset/
├── Sentinel-1/
│   ├── VV/
│   └── VH/
└── Sentinel-2/
    ├── NDVI/
    └── TrueColor/
```

Use the dashboard's preprocessing workflow before analysis. It validates readable files, normalises imagery, and preserves the source data.

## Deployment

The reference deployment separates the dashboard and API:

1. Deploy the FastAPI service to Render (or another Python host).
2. Attach persistent storage for the dataset, uploads, reports, and SQLite data.
3. Set `FORESTWATCH_ALLOWED_ORIGINS` to the dashboard URL.
4. Deploy `frontend/` to Vercel and set `VITE_FORESTWATCH_API` to the API URL.

## Responsible interpretation

ForestWatch provides monitoring signals, not a conclusive finding of deforestation. NDVI and image differences can be affected by seasonality, cloud cover, sensor differences, moisture, and image quality. Validate potential degradation with appropriate georeferenced data and field assessment before making management or conservation decisions.

## Contributing

Contributions and ideas are welcome.

1. Fork the repository.
2. Create a branch: `git checkout -b feature/my-improvement`
3. Commit your changes and open a pull request.

## License

Released under the [MIT License](LICENSE).
