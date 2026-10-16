"""
Data Preprocessing Pipeline for XAI Rainfall Prediction
========================================================
Handles:
  • Synthetic data generation (for development before real MOSDAC data)
  • Loading CSV / real satellite data
  • Cleaning, missing-value imputation, outlier clipping
  • Feature engineering (BT difference index)
  • Stratified train / val / test split
  • StandardScaler normalisation
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Windows terminals may not support the symbols used in progress messages.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow importing config from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data — realistic value ranges from INSAT-3D/3DR literature
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Create artificial satellite-style features for software development.

    Values are sampled within selected ranges. The coefficients in the
    illustrative target rule are not fitted to observations or validated as
    physical relationships. The binary class is sampled from its probability;
    rainfall_mm is generated afterward conditional on that class.
    """
    rng = np.random.RandomState(random_state)

    data = pd.DataFrame({
        'BT_TIR1':   rng.normal(280, 15, n_samples).clip(220, 320),
        'BT_TIR2':   rng.normal(275, 18, n_samples).clip(215, 315),
        'BT_WV':     rng.normal(245, 12, n_samples).clip(210, 270),
        'CTT':       rng.normal(260, 20, n_samples).clip(200, 310),
        'OLR':       rng.normal(220, 40, n_samples).clip(100, 320),
        'latitude':  rng.uniform(8.0, 35.0, n_samples),
        'longitude': rng.uniform(68.0, 97.0, n_samples),
        'month':     rng.choice([6, 7, 8, 9], n_samples),
        'hour':      rng.choice(range(24), n_samples),
    })

    # Key convection indicator
    data['BT_DIFF'] = data['BT_TIR1'] - data['BT_TIR2']

    # Logistic score driven by physical relationships
    z = (
        -0.30 * (data['CTT'] - 240) / 20
        - 0.25 * (data['OLR'] - 180) / 40
        + 0.20 * (data['BT_DIFF'] - 5) / 5
        - 0.15 * (data['BT_WV'] - 235) / 12
        + 0.10 * data['month'].isin([7, 8]).astype(float)
        + rng.normal(0, 0.3, n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    data['rainfall_class'] = (rng.random(n_samples) < prob).astype(int)

    # Reference rainfall amounts (mm/day) — not used as a feature
    data['rainfall_mm'] = np.where(
        data['rainfall_class'] == 1,
        rng.exponential(80, n_samples) + config.HEAVY_THRESHOLD,
        rng.exponential(15, n_samples),
    )

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load from CSV or fall back to synthetic data."""
    if filepath and os.path.exists(filepath):
        print(f"✔ Loading data from {filepath}")
        return pd.read_csv(filepath)

    print("⚠ No data file found — generating synthetic satellite data …")
    data = generate_synthetic_data()
    save_path = os.path.join(config.RAW_DATA_DIR, 'synthetic_satellite_data.csv')
    data.to_csv(save_path, index=False)
    print(f"  Saved → {save_path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning & feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input schema and ensure BT_DIFF exists."""
    df = data.copy()

    # Create BT_DIFF if absent
    if 'BT_DIFF' not in df.columns and {'BT_TIR1', 'BT_TIR2'}.issubset(df.columns):
        df['BT_DIFF'] = df['BT_TIR1'] - df['BT_TIR2']

    missing = [col for col in config.FEATURES + [config.TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")

    if df[config.TARGET].isnull().any():
        raise ValueError(f"{config.TARGET} cannot contain missing values")

    classes = set(pd.Series(df[config.TARGET]).dropna().unique())
    if not classes.issubset({0, 1}) or len(classes) != 2:
        raise ValueError(f"{config.TARGET} must contain both binary classes 0 and 1")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """Stratified split → train / validation / test."""
    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    val_ratio = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_tmp,
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Class distribution (train): {dict(y_train.value_counts())}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train, transform all splits."""
    scaler = StandardScaler()
    cols = X_train.columns

    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=cols, index=X_val.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=cols, index=X_test.index)

    return X_train_s, X_val_s, X_test_s, scaler


def clip_outliers_from_train(X_train, X_val, X_test):
    """Clip numeric features using bounds learned from training data only."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    numeric = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric:
        mean = X_train[col].mean()
        std = X_train[col].std()
        if pd.isna(std) or std == 0:
            continue
        lower, upper = mean - 3 * std, mean + 3 * std
        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)
    return X_train, X_val, X_test


def impute_missing_from_train(X_train, X_val, X_test):
    """Impute features using medians learned from the training split only."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    medians = X_train.median(numeric_only=True)
    all_missing = medians[medians.isna()].index.tolist()
    if all_missing:
        raise ValueError(f"Training data has entirely missing feature(s): {', '.join(all_missing)}")
    for col, median in medians.items():
        missing_count = int(X_train[col].isna().sum() + X_val[col].isna().sum() + X_test[col].isna().sum())
        if missing_count:
            print(f"  Filling {missing_count} missing values in {col} using the training median")
            X_train[col] = X_train[col].fillna(median)
            X_val[col] = X_val[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)
    return X_train, X_val, X_test


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str | None = None):
    """Execute the full preprocessing pipeline and persist artefacts."""
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    data = load_data(data_path)
    data = preprocess(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    X_train, X_val, X_test = impute_missing_from_train(X_train, X_val, X_test)
    X_train, X_val, X_test = clip_outliers_from_train(X_train, X_val, X_test)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # Persist
    for name, frame in [
        ('X_train', X_train_s), ('X_val', X_val_s), ('X_test', X_test_s),
        ('y_train', y_train),   ('y_val', y_val),   ('y_test', y_test),
    ]:
        frame.to_csv(os.path.join(config.PROCESSED_DATA_DIR, f'{name}.csv'), index=False)

    joblib.dump(scaler, os.path.join(config.MODELS_DIR, 'scaler.joblib'))
    print(f"\n✔ All artefacts saved to {config.PROCESSED_DATA_DIR}")
    print(f"✔ Scaler saved to {config.MODELS_DIR}/scaler.joblib")
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler


if __name__ == '__main__':
    run_pipeline()
