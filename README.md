# Explainable AI for Heavy-Rain Event Prediction

## Current scope

This major-project prototype demonstrates a machine-learning workflow for classifying generated satellite-style cases and explaining model behavior. It compares Random Forest, XGBoost (when installed), and a multilayer perceptron (MLP); uses SHAP for global and per-observation explanations; analyzes false positives and missed cases; and presents results in a Flask web application.

The current demo uses generated satellite-style features and an illustrative simulated binary label. The label is sampled from a logistic probability rule using CTT, OLR, BT_DIFF, BT_WV, month and random noise. Synthetic rainfall amounts are generated afterward conditional on the class; the class is **not** calculated by applying the configured 64.5 mm/day threshold to the generated amounts. The app does not ingest real INSAT-3D/3DR data, align satellite scans with rainfall observations, or predict a future rainfall window. The displayed scores only describe this synthetic experiment. They are not weather forecast skill and must not guide public warnings or safety decisions.

The intended research target is a three-hour nowcast from calibrated INSAT observations. A proposed event threshold is 60 mm accumulated over the next three hours (20 mm/hour average), subject to approval by the project guide. This future-window target is different from the current simulated class. The project needs real, time-matched data and a temporal or storm-based held-out evaluation before making any real-data performance claim.

## What the current system demonstrates

1. **Prediction:** the current prototype classifies a prepared synthetic observation as non-heavy or heavy from ten satellite-style features. The intended system will use recent satellite scans to predict rainfall during a defined future window. That temporal task is not implemented yet.
2. **Explainability:** SHAP global feature ranking and summary/dependence plots, plus per-observation feature contributions in the prediction page.
3. **Model evaluation:** accuracy, precision, recall, F1, ROC-AUC, and confusion matrices for the candidate models.
4. **Failure analysis:** false alarms and missed heavy-rain cases on the synthetic held-out set.
5. **Web interface:** a short overview, one-click demo, evidence page, and method/data page, with a JSON prediction endpoint.

## Repository layout

```text
config.py                         Shared paths, features, threshold, and settings
src/data_preprocessing.py         Data loading, synthetic generation, split, scaling
src/model_training.py             Model training, validation-based selection, evaluation
src/xai_explainer.py              SHAP reports and failure analysis
webapp/app.py                     Flask application and /api/predict endpoint
webapp/templates/                 HTML templates
webapp/static/css/style.css       Web styling
data/raw/                         Synthetic training data generated locally when needed
data/processed/                   Persisted train, validation, and test sets
models/                           Serialized models, scaler, and JSON reports
plots/                            Generated model and SHAP plots
```

## Setup

Use Python 3.10 or later. From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rebuild the project artifacts

Run these commands from the project root, in order:

```powershell
python -m src.data_preprocessing
python -m src.model_training
python -m src.xai_explainer
```

The preprocessing step writes scaled 70/15/15 train, validation, and test files under `data/processed/` and saves the scaler in `models/`. Missing-value medians and outlier-clipping bounds are learned from the training split and applied to validation and test data; the scaler is also fit on training data only. Training compares the three classifiers, chooses the model using validation F1, then reports final metrics on the held-out test split. The XAI step generates the global SHAP plots and failure report. Re-running preprocessing or training replaces the corresponding saved artifacts.

To train using a CSV instead of the synthetic fallback:

```powershell
python -c "from src.data_preprocessing import run_pipeline; run_pipeline(r'path\to\your_data.csv')"
```

The CSV must include the ten configured feature columns (or both TIR columns so `BT_DIFF` can be derived) and a binary `rainfall_class` target. For a real experiment, define that target from the agreed future rainfall window and threshold. The bundled synthetic target is a separate simulated label and must not be treated as rainfall ≥64.5 mm/day. The project currently expects satellite features and target to have already been calibrated, time-matched, and spatially matched. It does not perform HDF5/NetCDF ingestion or ground-truth matching.

## Start the web application

After artifacts exist, run:

```powershell
python -m webapp.app
```

Open `http://127.0.0.1:5000`. Choose **Run demo**, then click one of the prepared synthetic cases. The page shows the class, model score, and local SHAP contributions. Manual inputs remain under the project-review disclosure. **Evidence** combines synthetic benchmark metrics, global SHAP, and error counts. **Method & data** explains the intended system and steps for obtaining real data. Restart the app after rebuilding model artifacts.

Example JSON request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/predict `
  -Method Post -ContentType 'application/json' `
  -Body '{"BT_TIR1":250,"BT_TIR2":252,"BT_WV":245,"CTT":248,"OLR":210,"latitude":13.08,"longitude":80.27,"month":9,"hour":6}'
```

The API derives `BT_DIFF` if omitted. Inputs should be physically plausible and use the same units as the configured feature definitions. The API is a local project demonstration, not a production service.

## Understanding the reported scores

The current `models/metrics.json` and plots are rebuilt outputs from the bundled synthetic dataset. The model is selected using validation F1 and final metrics are calculated on the held-out test split. Metrics from synthetic data cannot establish performance on real weather events. To fulfill the nowcasting objective, prepare calibrated INSAT-3D/3DR observations paired with independent rainfall ground truth for a clearly defined future lead time and heavy-rain threshold. Split evaluation by independent time periods or storm events, then report data provenance, event coverage, recall, false alarms, and missed events. Accuracy alone can hide poor performance on rare heavy-rain events.

The feature contributions describe how the trained model responded to its input. They are not causal explanations of the atmosphere and do not guarantee that a prediction is correct.

## Deploy the demo

The repository includes a Render Blueprint (`render.yaml`) for a small hosted demo. Connect the repository to Render and deploy the Blueprint; it installs the pinned dependencies, starts Gunicorn, and generates a private Flask secret in the hosting environment. The hosted service runs the same synthetic demonstration and does not ingest user files or real satellite observations. Free hosting may sleep when idle, so the first visit can take a short while to wake.

For local use, the Flask development server is bound to `127.0.0.1` and debug mode is disabled. Do not change it to a public network server; use the included Gunicorn deployment configuration instead.
