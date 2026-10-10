import pytest
from motionscore.serial_telemetry import parse_telemetry_line


def test_parses_valid_sample():
    sample = parse_telemetry_line("MS1,1500,310\r\n")
    assert sample is not None
    assert sample["time_ms"] == 1500
    assert sample["adc_raw"] == 310
    assert sample["temperature_c"] == pytest.approx(24.98, abs=0.1)


def test_ignores_unrelated_or_malformed_lines():
    assert parse_telemetry_line("booting STM32") is None
    assert parse_telemetry_line("MS1,not-a-time,200") is None
    assert parse_telemetry_line("MS1,100,4096") is None


def test_uses_configured_reference_voltage():
    normal = parse_telemetry_line("MS1,100,310", vref_mv=3300.0)
    calibrated = parse_telemetry_line("MS1,100,310", vref_mv=3200.0)
    assert normal is not None and calibrated is not None
    assert calibrated["temperature_c"] < normal["temperature_c"]


def test_rejects_invalid_reference_voltage():
    with pytest.raises(ValueError):
        parse_telemetry_line("MS1,100,310", vref_mv=0)
