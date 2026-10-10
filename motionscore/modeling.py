"""Model training and evaluation utilities."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS

MODEL_PATH = Path("models/steady_state_model.joblib")


def train_model(data: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict:
    X = data[FEATURE_COLUMNS]
    y = data["steady_state_error"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)
    return {"model": model, "mae": mae, "rmse": rmse, "X_test": X_test,
            "y_test": y_test, "predictions": predictions}
