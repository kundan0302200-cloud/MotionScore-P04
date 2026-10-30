"""Wokwi RFC2217 receiver and bridge into the existing MotionScore model.

Potentiometer ADC is a simulated setpoint selector, NOT LM35 temperature.
The existing PI simulation creates the plant response; the existing feature
extractor and Random Forest are then used unchanged.
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
from .simulation import simulate_response


def parse_wokwi_frame(line: str) -> dict[str, int] | None:
    parts = line.strip().split(",")
    if len(parts) != 3 or parts[0] != "MS1":
        return None
    try:
        tick, raw = int(parts[1]), int(parts[2])
    except (TypeError, ValueError):
        return None
    if tick < 0 or not 0 <= raw <= 4095:
        return None
    return {"time_ms": tick, "adc_raw": raw}


def adc_to_simulated_setpoint(adc_raw: int, minimum: float = 20.0,
                              maximum: float = 50.0) -> float:
    if not 0 <= int(adc_raw) <= 4095:
        raise ValueError("adc_raw must be in [0, 4095].")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum.")
    return minimum + int(adc_raw) / 4095.0 * (maximum - minimum)


def build_prediction(model: Any, adc_raw: int, *, kp: float = 1.2,
                     ki: float = 0.35, plant_gain: float = 0.8,
                     time_constant: float = 1.5, disturbance: float = 1.0,
                     initial_offset: float = -15.0, noise_std: float = 0.03,
                     seed: int = 7, minimum_setpoint: float = 20.0,
                     maximum_setpoint: float = 50.0) -> dict[str, Any]:
    setpoint = adc_to_simulated_setpoint(adc_raw, minimum_setpoint, maximum_setpoint)
    response = simulate_response(
        setpoint, max(0.0, setpoint + initial_offset), kp, ki, plant_gain,
        time_constant, disturbance, noise_std=noise_std, seed=seed)
    features = extract_features(response, kp, ki, plant_gain, time_constant)
    frame = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]],
                         columns=FEATURE_COLUMNS)
    predicted_error = float(model.predict(frame)[0])
    reference_error = setpoint - float(response["true_output"].tail(20).mean())
    return {
        "adc_raw": int(adc_raw), "simulated_setpoint": float(setpoint),
        "features": features, "response": response,
        "predicted_steady_state_error": predicted_error,
        "predicted_steady_state_output": float(setpoint - predicted_error),
        "synthetic_reference_error": float(reference_error),
    }


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run 'python train.py' first.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Not a MotionScore model bundle.")
    stored = bundle.get("feature_columns")
    if stored is not None and list(stored) != FEATURE_COLUMNS:
        raise ValueError("Stored model feature columns do not match MotionScore.")
    model = bundle["model"]
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("Loaded model expects a different feature count.")
    return model


def capture_wokwi_predictions(url: str = "rfc2217://localhost:4000",
                              duration_seconds: float = 15.0,
                              baudrate: int = 115200, *,
                              model_path: str | Path = MODEL_PATH,
                              output_csv: str | Path | None = None,
                              minimum_setpoint: float = 20.0,
                              maximum_setpoint: float = 50.0) -> pd.DataFrame:
    if not url.strip() or duration_seconds <= 0 or baudrate <= 0:
        raise ValueError("Provide a URL, positive duration, and positive baudrate.")
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Install PySerial: python -m pip install pyserial") from exc
    model = load_model(model_path)
    records = []
    stop_at = time.monotonic() + duration_seconds
    with serial.serial_for_url(url.strip(), baudrate=baudrate, timeout=0.5) as conn:
        while time.monotonic() < stop_at:
            raw = conn.readline()
            if not raw:
                continue
            sample = parse_wokwi_frame(raw.decode("ascii", errors="replace"))
            if sample is None:
                continue
            pred = build_prediction(model, sample["adc_raw"],
                                    minimum_setpoint=minimum_setpoint,
                                    maximum_setpoint=maximum_setpoint)
            records.append({
                "time_ms": sample["time_ms"], "adc_raw": sample["adc_raw"],
                "simulated_setpoint": pred["simulated_setpoint"],
                "predicted_steady_state_error": pred["predicted_steady_state_error"],
                "predicted_steady_state_output": pred["predicted_steady_state_output"],
                "synthetic_reference_error": pred["synthetic_reference_error"],
            })
    columns = ["time_ms", "adc_raw", "simulated_setpoint",
               "predicted_steady_state_error", "predicted_steady_state_output",
               "synthetic_reference_error"]
    result = pd.DataFrame(records, columns=columns)
    if output_csv and not result.empty:
        result.to_csv(output_csv, index=False)
    return result


def run_receiver(url: str = "rfc2217://localhost:4000", baudrate: int = 115200,
                 *, model_path: str | Path = MODEL_PATH,
                 output_csv: str | Path | None = "wokwi_edge_predictions.csv") -> None:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Install PySerial: python -m pip install pyserial") from exc
    model = load_model(model_path)
    records = []
    print(f"Connecting to {url} at {baudrate} baud; keep Wokwi visible. Ctrl+C stops.")
    try:
        with serial.serial_for_url(url, baudrate=baudrate, timeout=1) as conn:
            while True:
                sample = parse_wokwi_frame(conn.readline().decode("ascii", errors="replace"))
                if sample is None:
                    continue
                pred = build_prediction(model, sample["adc_raw"])
                item = {k: pred[k] for k in (
                    "simulated_setpoint", "predicted_steady_state_error",
                    "predicted_steady_state_output", "synthetic_reference_error")}
                item.update(sample)
                records.append(item)
                print(json.dumps(item, separators=(",", ":")))
                if output_csv:
                    pd.DataFrame(records).to_csv(output_csv, index=False)
    except KeyboardInterrupt:
        print(f"\nReceiver stopped; {len(records)} valid frames received.")
        if output_csv and records:
            print(f"CSV saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="rfc2217://localhost:4000")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--csv", default="wokwi_edge_predictions.csv")
    args = parser.parse_args()
    run_receiver(args.url, args.baud, model_path=args.model, output_csv=args.csv)


if __name__ == "__main__":
    main()
