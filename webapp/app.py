"""
Flask Web Application — XAI Rainfall Prediction System
========================================================
Routes:
    /               — Home page
    /predict         — Prediction form + result
    /dashboard       — Model accuracy metrics & charts
    /explain         — SHAP explanations (global)
    /failure         — Model failure analysis
    /about           — Project methodology
    /api/predict     — JSON API for predictions
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, url_for

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.xai_explainer import explain_instance, get_explainer

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024  # JSON/form input only; no file uploads.


@app.after_request
def add_security_headers(response):
    """Set baseline browser protections for the public demonstration app."""
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    if request.is_secure:
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response

# ─── Load artefacts at startup ───────────────────────────────────────────────
MODEL = None
SCALER = None
METRICS = None
FAILURE_REPORT = None


def _load():
    global MODEL, SCALER, METRICS, FAILURE_REPORT
    model_path  = os.path.join(config.MODELS_DIR, 'best_model.joblib')
    scaler_path = os.path.join(config.MODELS_DIR, 'scaler.joblib')
    metrics_path = os.path.join(config.MODELS_DIR, 'metrics.json')
    failure_path = os.path.join(config.MODELS_DIR, 'failure_analysis.json')

    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            METRICS = json.load(f)
    if os.path.exists(failure_path):
        with open(failure_path) as f:
            FAILURE_REPORT = json.load(f)

_load()


# ─── Template helpers ────────────────────────────────────────────────────────

def _available_plots():
    """Gather all saved plot filenames."""
    plots = {}
    if os.path.isdir(config.PLOTS_DIR):
        for fname in os.listdir(config.PLOTS_DIR):
            if fname.endswith('.png'):
                plots[fname.replace('.png', '')] = fname
    return plots


def _parse_feature_value(feature, value):
    """Parse one prediction feature and return a concise validation message."""
    label = config.FEATURE_LABELS.get(feature, feature)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f'{label} must be a number.'
    if not math.isfinite(parsed):
        return None, f'{label} must be a finite number.'
    bounds = config.FEATURE_RANGES.get(feature)
    if bounds and not bounds[0] <= parsed <= bounds[1]:
        return None, f'{label} must be between {bounds[0]} and {bounds[1]}.'
    if feature in config.INTEGER_FEATURES and not parsed.is_integer():
        return None, f'{label} must be a whole number.'
    return parsed, None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metrics=METRICS)


@app.route('/healthz')
def healthz():
    """Hosting readiness check; report unhealthy if prediction artifacts are absent."""
    if MODEL is None or SCALER is None:
        return jsonify({'status': 'not_ready'}), 503
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    form_values = {}
    form_error = None

    if request.method == 'POST':
        # Collect form data
        raw = {}
        for feat in config.FEATURES:
            val = request.form.get(feat, '0')
            if feat == 'BT_DIFF' and not val.strip():
                form_values[feat] = ''
                continue
            parsed, error = _parse_feature_value(feat, val)
            if error:
                form_error = error
            else:
                raw[feat] = parsed
            form_values[feat] = val

        # Auto-compute BT_DIFF if not provided
        if 'BT_DIFF' not in request.form or not request.form['BT_DIFF']:
            raw['BT_DIFF'] = raw.get('BT_TIR1', 0) - raw.get('BT_TIR2', 0)
            form_values['BT_DIFF'] = str(raw['BT_DIFF'])

        if form_error is None and MODEL and SCALER:
            result = explain_instance(MODEL, SCALER, raw)
        elif form_error is None:
            form_error = 'Prediction artifacts are missing. Follow the README setup steps to build the model first.'

    return render_template('predict.html',
                           features=config.FEATURES,
                           feature_groups=[
                               ('Satellite and cloud features', config.FEATURES[:6]),
                               ('Observation context', config.FEATURES[6:]),
                           ],
                           labels=config.FEATURE_LABELS,
                           result=result,
                           form_error=form_error,
                           form_values=form_values)


@app.route('/dashboard')
def dashboard():
    plots = _available_plots()
    return render_template('dashboard.html', metrics=METRICS, plots=plots, report=FAILURE_REPORT)


@app.route('/explain')
def explain():
    plots = _available_plots()
    return render_template('explain.html', plots=plots)


@app.route('/failure')
def failure():
    plots = _available_plots()
    return render_template('failure.html', report=FAILURE_REPORT, plots=plots)


@app.route('/about')
def about():
    return render_template('about.html')


# ─── JSON API ────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if not MODEL or not SCALER:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Send a JSON object containing the observation features.'}), 400

    required = [feature for feature in config.FEATURES if feature != 'BT_DIFF']
    missing = [feature for feature in required if feature not in data]
    if missing:
        return jsonify({'error': f"Missing required feature(s): {', '.join(missing)}"}), 400

    raw = {}
    for feat in config.FEATURES:
        if feat == 'BT_DIFF' and feat not in data:
            continue
        value, error = _parse_feature_value(feat, data[feat])
        if error:
            return jsonify({'error': error}), 400
        raw[feat] = value

    if 'BT_DIFF' not in data:
        raw['BT_DIFF'] = raw.get('BT_TIR1', 0) - raw.get('BT_TIR2', 0)

    result = explain_instance(MODEL, SCALER, raw)
    return jsonify(result)


# ─── Static plot serving ─────────────────────────────────────────────────────

@app.route('/plots/<filename>')
def serve_plot(filename):
    from flask import send_from_directory
    return send_from_directory(config.PLOTS_DIR, filename)


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n🌧  XAI Rainfall Prediction — http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
