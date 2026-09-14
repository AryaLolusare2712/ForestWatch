# ForestWatch AI

ForestWatch AI is a local satellite-based forest-monitoring application built with Python, FastAPI, and Gradio. It helps users explore dated satellite observations, assess vegetation health, compare two dates for possible change, generate future NDVI estimates, manage alerts, and download monitoring reports.

## Features

- **Overview dashboard** — forest-health trend, latest observation, vegetation summary, and active alerts.
- **Satellite image upload** — add dated Sentinel-1 VV/VH or Sentinel-2 NDVI/true-colour images to the local dataset.
- **NDVI analysis** — view an NDVI scene and easy-to-read vegetation statistics.
- **Change detection** — choose any two available observation dates and download the change summary as CSV.
- **Potential degradation detection** — review lower-vegetation areas that may need field verification.
- **GIS visualisation** — explore available geographic and satellite information.
- **Future prediction** — automatically train the local prediction model from historical observations and estimate future NDVI values.
- **Alerts and reports** — manage alert status and generate weekly, monthly, or yearly downloadable reports.
- **Light and dark modes** — switch the Gradio dashboard appearance from the top-right toggle.

## Requirements

- Python 3.10 or newer
- A local satellite-image dataset (optional for launching the interface, required for analysis)

## Installation

Clone your GitHub repository and enter the project directory:

```powershell
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your local configuration file:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and set `FORESTWATCH_PROJECT_DATA` to the folder containing your satellite data. The supplied example values can be adjusted to suit your own data and analysis thresholds.

```ini
FORESTWATCH_DATASET=dataset
FORESTWATCH_OUTPUT=data
FORESTWATCH_PROJECT_DATA=C:\path\to\your\satellite-data
FORESTWATCH_LOW_NDVI=0.3
FORESTWATCH_HEALTHY_NDVI=0.6
FORESTWATCH_DECREASE=-0.15
```

## Run the application

Start the Gradio dashboard:

```powershell
python -m forestwatch.gradio_app
```

Open the local address shown in the terminal, normally `http://127.0.0.1:7860`.

To run the FastAPI service separately:

```powershell
python -m uvicorn forestwatch.api:app --reload
```

FastAPI interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Dataset and privacy

Satellite imagery, generated reports, models, local databases, and `.env` are intentionally excluded from GitHub by `.gitignore`. This keeps a repository small and prevents publishing local machine paths or private/project data.

Each user should provide their own source imagery and create their own `.env` file from `.env.example`. Uploaded images are added only to the local dataset folder; they are not sent anywhere by the application.

## Project structure

```text
forestwatch/
  api.py              FastAPI endpoints
  gradio_app.py       Gradio dashboard
  analysis.py         NDVI statistics and change analysis
  prediction.py       Historical-model training and forecasting
  alerts.py            Vegetation-health alerts
  reports.py           CSV and HTML report generation
  dataset_scanner.py  Local imagery discovery
tests/                Automated tests
```

## Important interpretation note

NDVI and change results are monitoring indicators, not proof of deforestation. Apparent change can be caused by seasonality, cloud cover, image artefacts, moisture, different sensors, or display-only imagery. Verify potential degradation with suitable georeferenced data and field assessment before making management decisions.

## License

Add the license that applies to your project before publishing or sharing it publicly.
