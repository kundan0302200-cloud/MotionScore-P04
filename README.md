# MotionScore — Steady-State Accuracy Predictor (P04)

A control-systems + machine-learning proof of concept that estimates the eventual steady-state error of a closed-loop PI-controlled plant from its early response. This P04 variant focuses on **steady-state accuracy, integral-action effects, and persistent bias**.

## Product preview

The Streamlit interface uses a dark glass-prism visual system with live control sliders, KPI cards, response/control-effort plots, a bias-threshold monitor, and an early-feature inspector.

## What it does

- Simulates a first-order plant with a PI controller and actuator saturation.
- Generates reproducible synthetic runs across setpoints, gains, plant dynamics, disturbances, and measurement noise.
- Extracts features from the first 30% of each run.
- Trains a Random Forest regressor to predict signed steady-state error.
- Reports held-out MAE and RMSE.
- Shows an interactive dashboard and a configurable persistent-bias threshold.

**Important:** The current dataset is synthetic and is intended for learning and demonstration. It is not validated for controlling or certifying a real plant.

## Run on Windows

From the project root in Command Prompt or PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

The local dashboard normally opens at http://localhost:8501. Keep the terminal open while using it.

On macOS/Linux, activate with `source .venv/bin/activate`.

## Architecture

```text
User / operator
   │ adjusts setpoint, controller gains and plant parameters
   ▼
Streamlit glass-prism dashboard (app.py)
   │
   ├── simulate_response() ──► synthetic PI + first-order plant time series
   │                                │
   │                                ▼
   ├── extract_features() ◄── first 30% of response
   │          │
   │          ▼
   ├── saved Random Forest model (models/steady_state_model.joblib)
   │          │
   │          ▼
   └── predicted signed steady-state error + bias-threshold indicator

Offline training:
generate_dataset() -> early-response feature table + final-error labels
-> train/test split -> RandomForestRegressor -> MAE/RMSE -> model artifact
```

## Project files

```text
app.py                         Streamlit UI and interactive charts
train.py                       Offline data generation, training and evaluation
motionscore/simulation.py      PI-controlled plant simulation and dataset creation
motionscore/features.py        Early-window feature extraction
motionscore/modeling.py        Random Forest training, evaluation and persistence
tests/test_features.py         Feature-shape and early-window tests
requirements.txt               Python dependencies
PROJECT_DOCUMENTATION.md       Architecture, viability, roadmap and viva handbook
models/steady_state_model.joblib  Generated after running train.py (not committed)
```

## Target and inputs

Target label:

`steady_state_error = setpoint - mean(final 20 plant-output samples)`

The model input features are:
- initial error and error at the early-window cutoff
- integral of error over the early window
- mean absolute early error
- mean and standard deviation of early control effort
- setpoint, proportional gain (Kp), integral gain (Ki), plant gain and plant time constant

The actual final output is used only to create training labels / an in-dashboard simulated reference; it is not a prediction feature.

## Model evaluation

Training prints mean absolute error (MAE) and root mean squared error (RMSE) on a held-out split. No fixed accuracy claim is made in advance: run the script and record the observed metrics from your own environment. For a serious evaluation, split by run/operating condition, add cross-validation, compare baselines, and validate against independently collected plant data.

## Limitations and responsible use

- Synthetic-data metrics do not establish real-world performance.
- Disturbance is not supplied directly as a feature; it must be inferred from the observed early response, which can limit identifiability.
- A single random split is only a first diagnostic, not a full validation protocol.
- Integral action often eliminates steady-state error in a stable, unsaturated loop, but actuator limits, disturbances, sensor bias, integral windup, and model mismatch can leave residual error.
- The UI’s bias flag is a threshold on the model prediction, not a safety alarm.
- Do not connect this proof of concept to hardware or use it for safety-critical decisions without engineering validation, uncertainty estimation, and appropriate safeguards.

## Tests

```powershell
python -m pytest -q
```

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the full system explanation, technology choices, module breakdown, technical viability, upgrade roadmap, demo flow, and viva questions.
