"""Parse MotionScore Edge serial frames from STM32 firmware.

Wire protocol: MS1,<time_ms>,<adc_raw>\r\n
Temperature is converted on the host using the configured ADC reference.
"""
from __future__ import annotations
import time
import pandas as pd


def parse_telemetry_line(line: str, vref_mv: float = 3300.0) -> dict | None:
    """Return one parsed sample, or None for malformed/unrelated serial lines."""
    if not (0.0 < vref_mv <= 3600.0):
        raise ValueError("vref_mv must be in (0, 3600].")
    parts = line.strip().split(",")
    if len(parts) != 3 or parts[0] != "MS1":
        return None
    try:
        time_ms = int(parts[1])
        adc_raw = int(parts[2])
    except (TypeError, ValueError):
        return None
    if time_ms < 0 or not 0 <= adc_raw <= 4095:
        return None
    voltage_mv = adc_raw * vref_mv / 4095.0
    # LM35's nominal scale is 10 mV per degree Celsius.
    temperature_c = voltage_mv / 10.0
    return {
        "time_ms": time_ms,
        "adc_raw": adc_raw,
        "voltage_mv": voltage_mv,
        "temperature_c": temperature_c,
    }


def capture_telemetry(
    port: str,
    duration_seconds: float = 10.0,
    baudrate: int = 115200,
    vref_mv: float = 3300.0,
    timeout_seconds: float = 0.2,
) -> pd.DataFrame:
    """Capture MotionScore Edge frames for a fixed time from a serial port."""
    if not port or not port.strip():
        raise ValueError("A serial port such as COM5 must be provided.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if baudrate <= 0:
        raise ValueError("baudrate must be positive.")

    import serial  # Import only when live telemetry is requested.

    rows = []
    stop_at = time.monotonic() + duration_seconds
    with serial.Serial(
        port.strip(), baudrate=baudrate, timeout=timeout_seconds,
        write_timeout=timeout_seconds
    ) as connection:
        connection.reset_input_buffer()
        while time.monotonic() < stop_at:
            raw = connection.readline()
            if not raw:
                continue
            parsed = parse_telemetry_line(
                raw.decode("ascii", errors="ignore"), vref_mv=vref_mv
            )
            if parsed is not None:
                rows.append(parsed)
    return pd.DataFrame(
        rows, columns=["time_ms", "adc_raw", "voltage_mv", "temperature_c"]
    )
