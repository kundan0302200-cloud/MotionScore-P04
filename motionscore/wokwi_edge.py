"""Wokwi RFC2217 receiver and MotionScore closed-loop prediction bridge.

Wokwi potentiometer readings are NOT LM35 measurements. The ADC value is
mapped to a user-selected simulated setpoint range, then used to parameterize
the existing synthetic PI plant. The trained model receives its original
11 early-response features only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, extract_features
from .modeling import MODEL_PATH
from .serial_telemetry import parse_telemetry_line
from .simulation import simulate_response


def adc_to_simulated_setpoint(
    adc_raw: int, minimum: float = 20.0, maximum: float = 50.0
) -> float:
    """Map a 12-bit potentiometer count to a simulated setpoint (not temperature)."""
    if not 0 <= int(adc_raw) <= 4095:
        raise ValueError("adc_raw must be in [0, 4095].")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum.")
    return minimum + (int(adc_raw) / 4095.0) * (maximum - minimum)


def build_prediction(
    model: Any,
    adc_raw: int,
    *,
    kp: float = 1.2,
    ki: float = 0.35,
    plant_gain: float = 0.8,
    time_constant: float = 1.5,
    disturbance: float = 1.0,
    initial_offset: float = -15.0,
    noise_std: float = 0.03,
    seed: int = 7,
) -> dict[str, Any]:
    """Turn one potentiometer sample into a synthetic PI response + ML prediction."""
    setpoint = adc_to_simulated_setpoint(adc_raw)
    initial_output = max(0.0, setpoint + initial_offset)
    response = simulate_response(
        setpoint=setpoint,
        initial_output=initial_output,
        kp=kp,
        ki=ki,
        plant_gain=plant_gain,
        time_constant=time_constant,
        disturbance=disturbance,
        noise_std=noise_std,
        seed=seed,
    )
    features = extract_features(response, kp, ki, plant_gain, time_constant)
    feature_frame = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    predicted_error = float(model.predict(feature_frame)[0])
    actual_error = setpoint - float(response["true_output"].tail(20).mean())
    return {
        "adc_raw": int(adc_raw),
        "simulated_setpoint": float(setpoint),
        "features": features,
        "predicted_steady_state_error": predicted_error,
        "predicted_steady_state_output": float(setpoint - predicted_error),
        "synthetic_reference_error": float(actual_error),
        "response": response,
    }


def load_model(model_path: str | Path = MODEL_PATH):
    """Load and validate the existing MotionScore model bundle."""
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python train.py' first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Model file is not a MotionScore model bundle.")
    stored = bundle.get("feature_columns")
    if stored is not None and list(stored) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match motionscore.features.FEATURE_COLUMNS."
        )
    model = bundle["model"]
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("Loaded model expects a different number of features.")
    return model


def run_receiver(
    url: str = "rfc2217://localhost:4000",
    baudrate: int = 115200,
    *,
    model_path: str | Path = MODEL_PATH,
    minimum_setpoint: float = 20.0,
    maximum_setpoint: float = 50.0,
    output_csv: str | Path | None = "wokwi_edge_predictions.csv",
) -> None:
    """Read MS1 frames from Wokwi's VS Code RFC2217 serial bridge."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Install PySerial: python -m pip install pyserial") from exc

    model = load_model(model_path)
    rows: list[dict[str, Any]] = []
    print(f"Connecting to {url} at {baudrate} baud...")
    print("Keep the Wokwi simulator tab visible and running. Press Ctrl+C to stop.")

    try:
        with serial.serial_for_url(url, baudrate=baudrate, timeout=1) as connection:
            while True:
                raw = connection.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                sample = parse_telemetry_line(line)
                if sample is None:
                    continue

                prediction = build_prediction(model, sample["adc_raw"])
                record = {
                    "received_at_unix": time.time(),
                    "mcu_time_ms": sample["time_ms"],
                    "adc_raw": sample["adc_raw"],
                    "simulated_setpoint": prediction["simulated_setpoint"],
                    "predicted_steady_state_error": prediction[
                        "predicted_steady_state_error"
                    ],
                    "predicted_steady_state_output": prediction[
                        "predicted_steady_state_output"
                    ],
                    "synthetic_reference_error": prediction["synthetic_reference_error"],
                }
                rows.append(record)
                print(json.dumps(record, separators=(",", ":")))

                if output_csv:
                    pd.DataFrame(rows).to_csv(output_csv, index=False)
    except KeyboardInterrupt:
        print("\nStopped receiver.")
        if output_csv and rows:
            print(f"Saved {len(rows)} records to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="rfc2217://localhost:4000")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--csv", default="wokwi_edge_predictions.csv")
    args = parser.parse_args()
    run_receiver(args.url, args.baud, model_path=args.model, output_csv=args.csv)


if __name__ == "__main__":
    main()
