# AQI Intelligence

A local AQI forecasting and model-evaluation platform for Anand Vihar, Delhi.

## Phase 1 status
This starter includes configuration, SQLAlchemy storage, OpenAQ sensor verification,
Open-Meteo historical/forecast clients, FastAPI health endpoints and a Streamlit shell.
It intentionally does not fabricate measurements, AQI values, model metrics or trained models.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `.env` and add a newly rotated OpenAQ API key. Never commit `.env`.

## Verify the provisional OpenAQ sensor IDs
```powershell
python -m app.data_collection.verify_openaq
```
The script checks each sensor's metadata, reports its parameter, unit, location and
coverage metadata, and flags sensors that do not belong to one Anand Vihar location.
It does not assume that the provisional IDs are valid.

## Run services
Terminal 1:
```powershell
uvicorn app.main:app --reload
```
Terminal 2:
```powershell
streamlit run app/dashboard/Home.py
```
API docs: http://127.0.0.1:8000/docs

## Important modelling safeguards
- Historical weather reanalysis is retrospective, not a forecast that was available live.
- For forecast-time weather features, use archived forecast runs or live forecasts with
  their issue timestamps; never substitute future observed weather.
- AQI is not inferred by applying AQI bands directly to PM2.5 concentrations.
- Official CPCB AQI needs the CPCB pollutant-specific averaging, completeness rules,
  breakpoints and maximum valid sub-index methodology.
- A real leaderboard requires predictions matched to subsequently observed targets.

## Next implementation phases
1. Verify OpenAQ station/sensor metadata and actual data coverage.
2. Collect and persist raw measurements and weather.
3. Preprocess, align, and create data-quality/EDA reports.
4. Implement leakage-safe targets, features and chronological ML training.
5. Implement forecast persistence, delayed actual matching and live leaderboard.
6. Complete dashboard pages and deterministic historical replay.
