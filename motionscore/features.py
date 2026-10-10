"""Feature extraction for early closed-loop response measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "initial_error", "early_error", "error_integral_early",
    "mean_abs_error_early", "control_mean_early", "control_std_early",
    "setpoint", "kp", "ki", "plant_gain", "time_constant",
]


def extract_features(
    response: pd.DataFrame,
    kp: float,
    ki: float,
    plant_gain: float,
    time_constant: float,
    early_fraction: float = 0.30,
) -> dict[str, float]:
    """Build model inputs from the early portion, not from final output."""
    n = max(5, int(len(response) * early_fraction))
    early = response.iloc[:n]
    errors = early["error"].to_numpy(dtype=float)
    times = early["time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return {
        "initial_error": float(errors[0]),
        "early_error": float(errors[-1]),
        "error_integral_early": float(np.trapz(errors, dx=dt)),
        "mean_abs_error_early": float(np.mean(np.abs(errors))),
        "control_mean_early": float(early["control"].mean()),
        "control_std_early": float(early["control"].std(ddof=0)),
        "setpoint": float(early["setpoint"].iloc[0]),
        "kp": float(kp),
        "ki": float(ki),
        "plant_gain": float(plant_gain),
        "time_constant": float(time_constant),
    }
