"""
Configuration for XAI Rainfall Prediction Project
All project-wide constants and paths live here.
"""
import os

# ─── Project paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Create directories automatically
for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Satellite feature columns ──────────────────────────────────────────────
FEATURES = [
    'BT_TIR1',       # Brightness Temperature — Thermal IR Channel 1 (K)
    'BT_TIR2',       # Brightness Temperature — Thermal IR Channel 2 (K)
    'BT_WV',          # Brightness Temperature — Water Vapor Channel (K)
    'CTT',            # Cloud Top Temperature (K)
    'OLR',            # Outgoing Longwave Radiation (W/m²)
    'BT_DIFF',        # BT_TIR1 − BT_TIR2 (derived split-window feature)
    'latitude',       # Latitude of observation
    'longitude',      # Longitude of observation
    'month',          # Month (1-12, captures seasonality)
    'hour',           # Hour of satellite observation (0-23)
]

TARGET = 'rainfall_class'   # Synthetic demo class; real labels must be built from future rainfall.

# Feature display names for the web UI
FEATURE_LABELS = {
    'BT_TIR1':   'Brightness Temp TIR-1 (K)',
    'BT_TIR2':   'Brightness Temp TIR-2 (K)',
    'BT_WV':     'Brightness Temp WV (K)',
    'CTT':       'Cloud Top Temp (K)',
    'OLR':       'Outgoing Longwave Radiation (W/m²)',
    'BT_DIFF':   'BT Difference TIR1−TIR2 (K)',
    'latitude':  'Latitude (°N)',
    'longitude': 'Longitude (°E)',
    'month':     'Month',
    'hour':      'Hour (UTC)',
}

# Basic physical/domain checks used by the web form and JSON endpoint.
FEATURE_RANGES = {
    'BT_TIR1': (100, 400),
    'BT_TIR2': (100, 400),
    'BT_WV': (100, 400),
    'CTT': (100, 400),
    'OLR': (0, 600),
    'BT_DIFF': (-200, 200),
    'latitude': (-90, 90),
    'longitude': (-180, 180),
    'month': (1, 12),
    'hour': (0, 23),
}
INTEGER_FEATURES = {'month', 'hour'}

# ─── IMD rainfall classification (mm / day) ─────────────────────────────────
RAINFALL_THRESHOLDS = {
    'light':          (0,       7.5),
    'moderate':       (7.5,     35.5),
    'rather_heavy':   (35.5,    64.5),
    'heavy':          (64.5,    115.5),
    'very_heavy':     (115.5,   204.5),
    'extremely_heavy':(204.5,   float('inf')),
}
HEAVY_THRESHOLD = 64.5   # mm/day reference used for synthetic amount sampling; does not assign the class.

# ─── Train / val / test split ratios ────────────────────────────────────────
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_STATE = 42

# ─── Web-app defaults ───────────────────────────────────────────────────────
FLASK_HOST  = os.environ.get('FLASK_HOST', '127.0.0.1')
FLASK_PORT  = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', '5000')))
FLASK_DEBUG = False
