import numpy as np
import pytest
from motionscore.features import FEATURE_COLUMNS
from motionscore.wokwi_edge import adc_to_simulated_setpoint, build_prediction, parse_wokwi_frame

class FakeRegressor:
    n_features_in_ = len(FEATURE_COLUMNS)
    def predict(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        assert frame.shape == (1, len(FEATURE_COLUMNS))
        return np.array([0.75])

def test_parses_ms1_frame():
    assert parse_wokwi_frame("MS1,1500,2050\\r\\n") == {"time_ms":1500,"adc_raw":2050}

def test_rejects_bad_frames():
    assert parse_wokwi_frame("boot") is None
    assert parse_wokwi_frame("MS1,x,100") is None
    assert parse_wokwi_frame("MS1,-1,100") is None
    assert parse_wokwi_frame("MS1,100,4096") is None

def test_adc_maps_to_setpoint_not_temperature():
    assert adc_to_simulated_setpoint(0) == pytest.approx(20.0)
    assert adc_to_simulated_setpoint(4095) == pytest.approx(50.0)
    assert adc_to_simulated_setpoint(2047) == pytest.approx(35.0, abs=0.01)

def test_model_receives_existing_feature_contract():
    result = build_prediction(FakeRegressor(), 2048)
    assert result["predicted_steady_state_error"] == pytest.approx(0.75)
    assert result["predicted_steady_state_output"] == pytest.approx(result["simulated_setpoint"]-0.75)
    assert list(result["features"]) == FEATURE_COLUMNS
    assert len(result["response"]) > 5

def test_invalid_adc_and_range_raise():
    with pytest.raises(ValueError):
        adc_to_simulated_setpoint(4096)
    with pytest.raises(ValueError):
        adc_to_simulated_setpoint(5, 50, 20)
