import pickle
import json
import os
import pandas as pd

MODEL_PATH = 'KNN.sav'
RANGES_PATH = 'feature_ranges.json'


def load_model() -> object:
    """Load the trained KNN model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def save_model(model: object) -> None:
    """Save the trained KNN model to disk."""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


def save_ranges(df: pd.DataFrame) -> None:
    """Compute and save feature ranges (min/max) for each feature."""
    ranges = {
        col: {'min': float(df[col].min()), 'max': float(df[col].max())}
        for col in df.columns if col != 'target'
    }
    with open(RANGES_PATH, 'w') as f:
        json.dump(ranges, f, indent=2)


def load_ranges() -> dict | None:
    """Load saved feature ranges, or return None if not available."""
    if not os.path.exists(RANGES_PATH):
        return None
    with open(RANGES_PATH, 'r') as f:
        return json.load(f)
