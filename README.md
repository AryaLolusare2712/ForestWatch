# ForestWatch

### Satellite-Based Forest Monitoring & Vegetation Change Prediction

ForestWatch is an intelligent forest-monitoring application that uses satellite imagery, vegetation indices, and machine learning to help users analyze forest health, identify potential vegetation degradation, and understand changes over time.

The platform combines a Gradio dashboard with a FastAPI backend to provide an interactive environment for satellite-image analysis, vegetation monitoring, forecasting, alerts, and report generation.

> **Project Status:** Active development

> **Application Type:** AI-powered geospatial monitoring

> **Interface:** Gradio

> **Backend:** FastAPI

---

## Overview

Forests play a vital role in maintaining biodiversity, regulating climate, and supporting ecosystems. Monitoring forest health over large areas can be challenging when relying solely on manual inspections.

ForestWatch aims to simplify this process by allowing users to explore historical satellite observations, analyze vegetation health using NDVI, compare observations from different dates, and generate insights that can support further investigation.

The system is designed as a local, extensible platform for experimenting with satellite-based environmental monitoring.

## Key Features

### Forest Health Overview

* View the latest available satellite observation.
* Explore vegetation-health summaries and historical trends.
* Review potential degradation indicators and active alerts.

### Satellite Image Analysis

* Work with locally stored Sentinel-1 and Sentinel-2 imagery.
* Upload dated satellite observations through the dashboard.
* Preprocess source imagery non-destructively before analysis.
* Explore NDVI and true-color imagery.

### NDVI-Based Vegetation Monitoring

* Calculate and visualize vegetation-related statistics.
* Examine vegetation-health patterns across observations.
* Identify areas with potentially reduced vegetation health.

### Change Detection

* Compare satellite observations from two dates.
* Analyze changes in vegetation-related indicators.
* Export change summaries for further analysis.

### Vegetation Prediction

* Train a local prediction model using historical observations.
* Generate future NDVI estimates based on available historical data.
* Explore possible vegetation trends over time.

### GIS & Multi-Satellite Visualization

* Explore satellite observations in a geospatial monitoring workflow.
* Support analysis involving multiple satellite data sources.
* Provide a foundation for future map-based environmental intelligence.

### Alerts & Reports

* Review and manage vegetation-health alerts.
* Generate weekly, monthly, and yearly monitoring reports.
* Export analysis results for documentation and review.

### Interactive Dashboard

* Gradio-based user interface.
* Light and dark appearance modes.
* Interactive visualizations and monitoring controls.

---

## How It Works

```text
Satellite Imagery
       │
       ▼
Local Dataset Scanner
       │
       ▼
Non-destructive Image Preprocessing
       │
       ▼
Image Processing & NDVI Analysis
       │
       ▼
Vegetation Health Assessment
       │
       ├──────────────► Change Detection
       │
       ├──────────────► Alert Generation
       │
       ├──────────────► Historical Prediction
       │
       └──────────────► Monitoring Reports
                              │
                              ▼
                    Gradio Dashboard
                              │
                              ▼
                         User Insights
```

---

## Technology Stack

| Technology    | Purpose                                |
| ------------- | -------------------------------------- |
| Python        | Core application and analysis logic    |
| Gradio        | Interactive monitoring dashboard       |
| FastAPI       | Backend API services                   |
| Uvicorn       | ASGI server                            |
| NumPy         | Numerical computation                  |
| Pandas        | Data processing and analysis           |
| Pillow        | Image handling                         |
| Rasterio      | Raster and geospatial image processing |
| Plotly        | Interactive visualizations             |
| Matplotlib    | Data visualization                     |
| Scikit-learn  | Machine learning and prediction        |
| SQLAlchemy    | Database interaction                   |
| python-dotenv | Environment configuration              |

---

## Project Structure

```text
Satellite-main/
│
├── forestwatch/
│   ├── __init__.py
│   ├── api.py
│   ├── gradio_app.py
│   ├── analysis.py
│   ├── prediction.py
│   ├── alerts.py
│   ├── reports.py
│   ├── database.py
│   ├── config.py
│   ├── schemas.py
│   ├── dataset_scanner.py
│   ├── preprocessing.py
│   └── multisatellite.py
│
├── dataset/
│   ├── Sentinel-1/
│   └── Sentinel-2/
│
├── tests/
│
├── requirements.txt
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AryaLolusare2712/ForestWatch-AI.git
cd ForestWatch-AI
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local configuration file:
Create `.env` and configure the dataset and output directories.

Example:

```env
FORESTWATCH_DATASET=dataset
FORESTWATCH_OUTPUT=data
FORESTWATCH_PROJECT_DATA=dataset
FORESTWATCH_LOW_NDVI=0.3
FORESTWATCH_HEALTHY_NDVI=0.6
FORESTWATCH_DECREASE=-0.15
```

> Adjust paths and thresholds according to your local dataset and project configuration. Keep private credentials and machine-specific settings out of GitHub.

---

## Running the Application

### Start the Gradio Dashboard

```bash
python -m forestwatch.gradio_app
```

Open the local URL displayed in your terminal, typically:

```text
http://127.0.0.1:7860
```

### Start the FastAPI Backend

In a separate terminal:

```bash
python -m uvicorn forestwatch.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the React GIS Dashboard

The standalone React/Vite GIS dashboard is in `frontend/`. It uses FastAPI for dated NDVI change values, so start the API first. Then, in a second terminal with Node.js 20+ installed:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite address shown in the terminal, normally `http://127.0.0.1:5173`. Select an earlier and later observation date to update the GIS period. The dashboard reports actual NDVI-index change; it does not scale values to −100% or +100%.

### Login and email alerts

The React dashboard has a local registration and login page. Its email address is the destination for newly created vegetation-health alerts. To enable delivery, fill the `FORESTWATCH_SMTP_*` values in your private `.env` file. Use an application password from your email provider, not your normal account password. If SMTP is not configured, alerts remain visible in the dashboard and no email is sent.

### CNN and YOLO comparison

The **CNN vs YOLO Comparison** tab creates a controlled experiment using the supplied 2024–2025 Gorewada aligned temporal cube and high-confidence change label. It trains a small CNN for patch classification and YOLOv8n for change-region detection, then reports held-out spatial-tile metrics. This dataset has only one labelled period, so the comparison is useful for experimentation only—not an operational accuracy claim. Add independent labelled dates before choosing a model for deployment.

---

## Dataset

ForestWatch is designed to work with locally available satellite observations, including:

* **Sentinel-1:** VV/VH radar imagery for multi-satellite monitoring workflows.
* **Sentinel-2:** NDVI and true-color imagery for vegetation analysis.

The repository may contain sample imagery for development and demonstration. For larger experiments, users can configure their own dataset location through `.env`.

### Suggested Dataset Organization

```text
dataset/
├── Sentinel-1/
│   ├── VV/
│   └── VH/
│
└── Sentinel-2/
    ├── NDVI/
    └── TrueColor/
```

Use dated filenames or metadata so observations can be organized chronologically.

### Preprocess before analysis

Before using NDVI analysis, change detection, or prediction, open **Satellite Image Upload** in the dashboard and select **Preprocess local satellite data**. This validates readable image files, corrects EXIF orientation, standardizes images to three RGB channels, and normalizes the pixels to a 0–1 range.

The source satellite files are never modified. A local processing manifest is written to `data/processed/` so you can review which images were prepared and which unreadable files were skipped.

The bundled Gorewada reference boundary is also used as a no-data mask for scenes on the matching 1500 × 1224 pixel grid. Black pixels outside the boundary, or black gaps inside it, are excluded from NDVI and change calculations rather than being counted as low vegetation.

---

## Example Use Cases

1. **Historical Forest Monitoring** — Examine vegetation health across available observation dates.
2. **Potential Degradation Screening** — Identify areas requiring additional investigation.
3. **Vegetation Trend Analysis** — Explore changes in NDVI over time.
4. **Satellite Data Comparison** — Work with optical and radar observations in a unified workflow.
5. **Environmental Reporting** — Generate monitoring reports for documentation and analysis.

---

## Future Enhancements

* [ ] Advanced geospatial map integration.
* [ ] Automated satellite data acquisition.
* [ ] Improved cloud and image-quality filtering.
* [ ] More robust spatial change-detection models.
* [ ] Multi-region forest monitoring.
* [ ] Drone imagery integration for field-level verification.
* [ ] Enhanced machine learning models for vegetation forecasting.
* [ ] Cloud deployment and scalable processing.
* [ ] Role-based access and collaborative monitoring.

---

## Important Interpretation Note

ForestWatch provides satellite-based monitoring indicators and experimental predictions. A decrease in NDVI or an apparent change in imagery does not, by itself, prove deforestation.

Results may be influenced by seasonality, cloud cover, image quality, moisture, sensor differences, and other environmental factors. Potential degradation should be verified using appropriate georeferenced data and field assessment before making management or conservation decisions.

---

## Privacy & Data Handling

* Satellite imagery is processed using the project's local dataset workflow.
* Generated reports, local databases, and model outputs are intended for local project use.
* Do not commit `.env` files containing secrets or private machine paths.
* Review `.gitignore` before publishing the repository.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Commit your changes.

```bash
git commit -m "Add your feature"
```

4. Push the branch.

```bash
git push origin feature/your-feature
```

5. Open a pull request.

---
