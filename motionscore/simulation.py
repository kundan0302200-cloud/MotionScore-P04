"""Synthetic PI-controlled first-order plant response generator."""
from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_response(setpoint: float, initial_output: float, kp: float, ki: float,
                      plant_gain: float, time_constant: float, disturbance: float,
                      noise_std: float = 0.0, duration: float = 12.0,
                      dt: float = 0.05, seed: int | None = None) -> pd.DataFrame:
    """Simulate a first-order plant with PI control and actuator saturation."""
    if dt <= 0 or duration <= 0 or time_constant <= 0 or plant_gain < 0:
        raise ValueError("dt, duration and time_constant must be positive; plant_gain must be non-negative.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration + dt / 2, dt)
    y = np.zeros_like(t); u = np.zeros_like(t); error = np.zeros_like(t)
    integral = 0.0
    y[0] = initial_output
    for i in range(1, len(t)):
        error[i - 1] = setpoint - y[i - 1]
        integral += error[i - 1] * dt
        u[i - 1] = np.clip(kp * error[i - 1] + ki * integral, -100.0, 100.0)
        dy = (-y[i - 1] + plant_gain * u[i - 1] + disturbance) / time_constant
        y[i] = y[i - 1] + dy * dt
    error[-1] = setpoint - y[-1]
    u[-1] = u[-2] if len(u) > 1 else 0.0
    measured_y = y + rng.normal(0.0, noise_std, size=len(y))
    return pd.DataFrame({"time": t, "setpoint": setpoint, "output": measured_y,
                         "true_output": y, "error": setpoint - measured_y, "control": u})

def generate_dataset(n_runs: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic response runs with early features and final-error labels."""
    from .features import extract_features
    rng = np.random.default_rng(seed)
    rows = []
    for run_id in range(n_runs):
        setpoint = float(rng.uniform(5, 50))
        initial_output = float(setpoint + rng.uniform(-20, 20))
        kp = float(rng.uniform(0.2, 3.0)); ki = float(rng.uniform(0.0, 1.2))
        plant_gain = float(rng.uniform(0.3, 1.5))
        time_constant = float(rng.uniform(0.5, 4.0))
        disturbance = float(rng.uniform(-3.0, 3.0))
        noise_std = float(rng.uniform(0.0, 0.15))
        response = simulate_response(setpoint, initial_output, kp, ki, plant_gain,
                                     time_constant, disturbance, noise_std,
                                     seed=int(rng.integers(0, 2**31 - 1)))
        features = extract_features(response, kp, ki, plant_gain, time_constant)
        # Final settled output is a label, never an input feature.
        features["steady_state_error"] = setpoint - float(response["true_output"].tail(20).mean())
        features["run_id"] = run_id
        rows.append(features)
    return pd.DataFrame(rows)
