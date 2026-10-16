# Explainable AI for Heavy-Rain Event Prediction

[Open the live demo](https://rainfall-lab-demo.onrender.com/) · [Run a sample prediction](https://rainfall-lab-demo.onrender.com/predict)

> Demo note: this public prototype uses simulated satellite-style inputs; it does not yet ingest live INSAT data or provide an operational forecast.

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
