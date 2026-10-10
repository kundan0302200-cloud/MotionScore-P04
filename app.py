"""Streamlit dashboard for MotionScore."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from motionscore.simulation import simulate_response
from motionscore.features import extract_features, FEATURE_COLUMNS
from motionscore.modeling import MODEL_PATH

st.set_page_config(page_title="MotionScore | P04", page_icon="📈", layout="wide")
st.title("MotionScore")
st.subheader("P04 — Steady-State Accuracy Predictor")
st.write(
    "Predict the final steady-state error from the early response of a "
    "closed-loop PI-controlled system. This demo uses a synthetic plant."
)

if not Path(MODEL_PATH).exists():
    st.warning("No trained model found. Run `python train.py` first.")
    st.code("pip install -r requirements.txt\npython train.py\nstreamlit run app.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

with st.sidebar:
    st.header("Control parameters")
    setpoint = st.slider("Target / setpoint", 5.0, 50.0, 25.0)
    initial_output = st.slider("Initial output", 0.0, 60.0, 10.0)
    kp = st.slider("Proportional gain (Kp)", 0.2, 3.0, 1.2)
    ki = st.slider("Integral gain (Ki)", 0.0, 1.2, 0.35)
    plant_gain = st.slider("Plant gain", 0.3, 1.5, 0.8)
    time_constant = st.slider("Time constant", 0.5, 4.0, 1.5)
    disturbance = st.slider("Constant disturbance", -3.0, 3.0, 1.0)
    threshold = st.slider("Bias alert threshold", 0.1, 5.0, 1.0)

response = simulate_response(
    setpoint, initial_output, kp, ki, plant_gain, time_constant,
    disturbance, noise_std=0.03, seed=7
)
features = extract_features(response, kp, ki, plant_gain, time_constant)
X = [[features[col] for col in FEATURE_COLUMNS]]
predicted_error = float(model.predict(X)[0])
predicted_output = setpoint - predicted_error
actual_final_output = float(response["true_output"].tail(20).mean())
actual_error = setpoint - actual_final_output
bias_detected = abs(predicted_error) >= threshold

c1, c2, c3 = st.columns(3)
c1.metric("Predicted steady-state error", f"{predicted_error:.3f}")
c2.metric("Predicted final output", f"{predicted_output:.3f}")
c3.metric("Simulated final error (reference)", f"{actual_error:.3f}")

if bias_detected:
    st.warning("Potential persistent bias: predicted absolute final error exceeds the selected threshold.")
else:
    st.success("Predicted final error is within the selected threshold.")

fig, ax = plt.subplots()
ax.plot(response["time"], response["output"], label="Measured output")
ax.plot(response["time"], response["setpoint"], "--", label="Setpoint")
ax.axvline(response["time"].iloc[int(len(response) * 0.30)], linestyle=":", label="Prediction cutoff")
ax.set_xlabel("Time")
ax.set_ylabel("Output")
ax.set_title("Closed-loop response")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "The simulated final error is shown only as a reference for this demo. "
    "The model prediction uses early-response features, not the final output."
)
with st.expander("Inspect early-response features"):
    st.json(features)
