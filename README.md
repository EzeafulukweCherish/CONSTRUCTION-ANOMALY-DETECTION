# Construction Project Fraud Detector

An earned-value (EVM) baseline, an Isolation Forest consistency checker that
compares reported labor/material effort against verified drone-based
completion, and an adaptive stress test that simulates a fraudster who gets
stealthier over time against a static vs. a retraining detector.

Built for the [BIM-AI Integrated Dataset](https://www.kaggle.com/datasets/ziya07/bim-ai-integrated-dataset)
(or any CSV with matching columns: `Project_Type`, `Location`,
`Weather_Condition`, `Planned_Cost`, `Actual_Cost`, `Planned_Duration`,
`Actual_Duration`, `Completion_Percentage`, plus structural/environmental/
safety columns).

## Files

- `app.py` — the full Streamlit app (data upload, cleaning, EVM, model, stress test)
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build
- `docker-compose.yml` — one-command local run via Docker
- `.dockerignore` — keeps the image lean

## Run locally (no Docker)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501) and
upload your CSV in the app.

## Run with Docker

```bash
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```

Or with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501.

## Notes for the demo

- The dataset's raw labor/material/cost columns are not correlated with each
  other (verified during EDA), so the app rebuilds realistic labor and
  material figures scaled to verified completion, then secretly tampers with
  10% of projects to create known fraud cases for testing. This is disclosed
  in-app and should be disclosed in any presentation — it is a legitimate
  simulation, not real fraud data.
- The "Stress test" tab is stochastic per session (seeded), so re-running it
  will give the same sequence within a session but can differ across app
  restarts.
- `is_fraud` is only visible in the UI because this is simulated ground
  truth used to evaluate the model. A production version would not have
  this column.
