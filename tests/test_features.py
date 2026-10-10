import numpy as np
from motionscore.simulation import simulate_response
from motionscore.features import extract_features, FEATURE_COLUMNS

def test_feature_extraction_has_expected_columns():
    response = simulate_response(10, 2, 1.0, 0.3, 0.8, 1.5, 0.5, seed=1)
    features = extract_features(response, 1.0, 0.3, 0.8, 1.5)
    assert list(features.keys()) == FEATURE_COLUMNS
    assert all(np.isfinite(value) for value in features.values())

def test_early_features_ignore_late_response_changes():
    response = simulate_response(10, 2, 1.0, 0.3, 0.8, 1.5, 0.5, seed=1)
    first = extract_features(response, 1.0, 0.3, 0.8, 1.5)
    response.loc[response.index[-20:], "error"] -= 1000
    response.loc[response.index[-20:], "output"] += 1000
    second = extract_features(response, 1.0, 0.3, 0.8, 1.5)
    assert first == second
