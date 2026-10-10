"""Glass-prism Streamlit dashboard for MotionScore P04."""
from pathlib import Path
import joblib
import plotly.graph_objects as go
import streamlit as st

from motionscore.simulation import simulate_response
from motionscore.features import extract_features, FEATURE_COLUMNS
from motionscore.modeling import MODEL_PATH

st.set_page_config(
    page_title="MotionScore | Steady-State Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium glass-prism visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#070b16; --panel:rgba(16,25,47,.70); --line:rgba(171,192,255,.17);
        --muted:#9aa8c7; --white:#f4f7ff; --cyan:#67e8f9; --violet:#a78bfa; --mint:#6ee7b7; }
html, body, [class*="css"] { font-family:'Manrope', sans-serif; }
.stApp {
 background:
  radial-gradient(ellipse at 12% 0%, rgba(72,75,180,.26), transparent 34%),
  radial-gradient(ellipse at 90% 12%, rgba(30,166,196,.16), transparent 29%),
  radial-gradient(ellipse at 70% 90%, rgba(128,72,205,.12), transparent 34%),
  #070b16;
 color:var(--white);
}
[data-testid="stHeader"] { background:rgba(7,11,22,.68); }
[data-testid="stToolbar"] { right:1rem; }
.block-container { max-width:1450px; padding-top:2.1rem; padding-bottom:3.5rem; }
section[data-testid="stSidebar"] {
 background:linear-gradient(160deg,rgba(18,27,52,.94),rgba(9,15,30,.96));
 border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] * { color:#e9efff; }
section[data-testid="stSidebar"] [data-testid="stSlider"] { padding-top:.2rem; }
h1,h2,h3 { letter-spacing:-.035em; }
p, li { color:#c4cee5; }
a { color:var(--cyan)!important; }
[data-testid="stMetric"] {
 background:linear-gradient(135deg,rgba(27,42,73,.80),rgba(14,22,42,.68));
 border:1px solid var(--line); border-radius:18px; padding:18px 20px;
 box-shadow:0 16px 40px rgba(0,0,0,.16); backdrop-filter:blur(18px);
}
[data-testid="stMetricLabel"] { color:#a9b8d7!important; font-size:.78rem!important; }
[data-testid="stMetricValue"] { color:#f7f9ff!important; font-weight:800; }
div[data-testid="stAlert"] { border-radius:14px; border:1px solid var(--line); }
.stButton button { border-radius:12px; border:1px solid rgba(103,232,249,.4); }
.glass {
 background:linear-gradient(135deg,rgba(31,46,80,.76),rgba(13,21,40,.70));
 border:1px solid rgba(171,192,255,.18); border-radius:22px;
 padding:22px 24px; box-shadow:0 18px 48px rgba(0,0,0,.20);
 backdrop-filter:blur(20px); position:relative; overflow:hidden;
}
.glass:before {
 content:""; position:absolute; top:0; left:0; right:0; height:1px;
 background:linear-gradient(90deg,transparent,rgba(103,232,249,.7),rgba(167,139,250,.6),transparent);
}
.hero { padding:28px 30px 25px; margin-bottom:20px; position:relative; overflow:hidden; }
.hero:after {
 content:""; position:absolute; width:360px; height:360px; right:-80px; top:-210px;
 background:conic-gradient(from 30deg,rgba(103,232,249,.15),rgba(167,139,250,.20),transparent,rgba(103,232,249,.10));
 filter:blur(3px); transform:rotate(18deg); pointer-events:none;
}
.eyebrow { color:var(--cyan); font-family:'DM Mono',monospace; letter-spacing:.17em;
 text-transform:uppercase; font-size:.70rem; margin-bottom:10px; }
.brand { font-size:clamp(2.4rem,4.5vw,4.3rem); font-weight:800; letter-spacing:-.075em;
 line-height:1.0; margin:0; color:#f8faff; }
.brand span { background:linear-gradient(100deg,#67e8f9 10%,#a78bfa 58%,#6ee7b7 100%);
 -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { max-width:760px; margin:14px 0 0; font-size:1rem; color:#bdc9e4; line-height:1.7; }
.pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.pill { display:inline-block; font-family:'DM Mono',monospace; font-size:.68rem;
 color:#d7e8ff; background:rgba(103,232,249,.07); border:1px solid rgba(103,232,249,.18);
 padding:7px 10px; border-radius:999px; }
.section-kicker { color:#7dd3fc; font-family:'DM Mono',monospace; text-transform:uppercase;
 letter-spacing:.16em; font-size:.69rem; margin:8px 0 7px; }
.section-title { color:#f0f5ff; font-size:1.35rem; font-weight:800; margin:0 0 14px; letter-spacing:-.04em; }
.small-muted { color:#9aa8c7; font-size:.84rem; line-height:1.65; }
.stat-label { color:#9aa8c7; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }
.stat-number { font-size:1.8rem; font-weight:800; letter-spacing:-.045em; color:#f7f9ff; margin-top:4px; }
.stat-note { color:#aab9d8; font-size:.8rem; margin-top:4px; }
.status { display:inline-block; border-radius:999px; padding:6px 10px; font-size:.72rem;
 font-family:'DM Mono',monospace; }
.status-good { color:#a7f3d0; border:1px solid rgba(110,231,183,.35); background:rgba(16,185,129,.10); }
.status-warn { color:#fde68a; border:1px solid rgba(251,191,36,.35); background:rgba(245,158,11,.10); }
div[data-testid="stPlotlyChart"] { border:1px solid rgba(171,192,255,.13); border-radius:18px; overflow:hidden; }
hr { border-color:rgba(171,192,255,.13)!important; }
footer { visibility:hidden; }
@media (max-width:760px) {
 .block-container { padding-left:1rem; padding-right:1rem; }
 .hero { padding:22px 20px; }
 .glass { padding:18px; border-radius:17px; }
}
</style>
""", unsafe_allow_html=True)

model_path = Path(MODEL_PATH)
if not model_path.is_absolute():
    model_path = Path(__file__).resolve().parent / model_path

# ---------- Hero ----------
st.markdown("""
<div class="glass hero">
  <div class="eyebrow">P04 / Closed-loop intelligence</div>
  <h1 class="brand">Motion<span>Score</span></h1>
  <p class="hero-sub">Steady-State Accuracy Predictor — estimate the residual error a controlled system may retain after settling. Explore how PI control, plant dynamics and disturbances shape the response.</p>
  <div class="pill-row">
    <span class="pill">◈ GLASS PRISM UI</span>
    <span class="pill">RANDOM FOREST REGRESSION</span>
    <span class="pill">SYNTHETIC PLANT LAB</span>
    <span class="pill">PERSISTENT BIAS CHECK</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_path.exists():
    st.markdown("""
    <div class="glass">
      <div class="section-kicker">Model setup required</div>
      <div class="section-title">Train the predictor to unlock the lab</div>
      <p class="small-muted">The trained model file was not found. From the project root, create your environment, install requirements, then train the model.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("python -m pip install -r requirements.txt\npython train.py\npython -m streamlit run app.py", language="bash")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### ◈ Control lab")
    st.caption("Tune the simulated plant and controller.")
    st.markdown("---")
    setpoint = st.slider("Target / setpoint", 5.0, 50.0, 25.0, 0.5)
    initial_output = st.slider("Initial output", 0.0, 60.0, 10.0, 0.5)
    st.markdown("**Controller gains**")
    kp = st.slider("Proportional gain · Kp", 0.2, 3.0, 1.2, 0.05)
    ki = st.slider("Integral gain · Ki", 0.0, 1.2, 0.35, 0.05)
    st.markdown("**Plant dynamics**")
    plant_gain = st.slider("Plant gain", 0.3, 1.5, 0.8, 0.05)
    time_constant = st.slider("Time constant · τ", 0.5, 4.0, 1.5, 0.1)
    disturbance = st.slider("Constant disturbance", -3.0, 3.0, 1.0, 0.1)
    st.markdown("**Bias monitor**")
    threshold = st.slider("Bias alert threshold", 0.1, 5.0, 1.0, 0.1)
    st.markdown("---")
    st.markdown("<span class='pill'>SIMULATION MODE · NOT HARDWARE CONNECTED</span>", unsafe_allow_html=True)
    st.caption("Inputs are illustrative. Validate with measured plant data before engineering use.")

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
early_error = abs(features["early_error"])
initial_error = abs(features["initial_error"])
reduction = 100 * (1 - early_error / initial_error) if initial_error > 1e-9 else 0.0

# ---------- KPI row ----------
st.markdown('<div class="section-kicker">Live model readout</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted final error", f"{predicted_error:+.3f}", help="Target minus predicted settled output.")
k2.metric("Predicted settled output", f"{predicted_output:.3f}")
k3.metric("Reference final error", f"{actual_error:+.3f}", help="Known end-of-run value from the synthetic simulation; not used by the model.")
k4.metric("Early error reduction", f"{reduction:.1f}%")

if bias_detected:
    st.markdown(f"<span class='status status-warn'>⚠ PERSISTENT-BIAS WATCH</span> &nbsp; <span class='small-muted'>Absolute predicted error is at or above the selected threshold ({threshold:.2f}).</span>", unsafe_allow_html=True)
else:
    st.markdown(f"<span class='status status-good'>✓ WITHIN THRESHOLD</span> &nbsp; <span class='small-muted'>Predicted absolute error is below the selected threshold ({threshold:.2f}).</span>", unsafe_allow_html=True)

st.markdown("---")
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-kicker">01 / Dynamic response</div><div class="section-title">Output tracking over time</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=response["time"], y=response["output"], mode="lines",
        name="Measured output", line=dict(color="#67e8f9", width=2.5),
        fill="tozeroy", fillcolor="rgba(103,232,249,0.055)"))
    fig.add_trace(go.Scatter(x=response["time"], y=response["setpoint"], mode="lines",
        name="Setpoint", line=dict(color="#a78bfa", width=2, dash="dash")))
    cutoff = float(response["time"].iloc[min(len(response)-1, int(len(response)*0.30))])
    fig.add_vline(x=cutoff, line_width=1.5, line_dash="dot", line_color="#fbbf24",
                  annotation_text="prediction cutoff", annotation_font_color="#fbbf24")
    fig.update_layout(
        height=390, margin=dict(l=10,r=15,t=12,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,12,25,.32)",
        font=dict(family="Manrope",color="#dce7ff",size=11),
        xaxis=dict(title="Time (s)", gridcolor="rgba(171,192,255,.10)", zeroline=False),
        yaxis=dict(title="Output", gridcolor="rgba(171,192,255,.10)", zeroline=False),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

with right:
    st.markdown('<div class="section-kicker">02 / Diagnostic summary</div><div class="section-title">Steady-state outlook</div>', unsafe_allow_html=True)
    label = "Potential residual bias" if bias_detected else "Within chosen tolerance"
    css = "status-warn" if bias_detected else "status-good"
    st.markdown(f"""
    <div class="glass">
      <div class="stat-label">Bias monitor</div>
      <div style="margin:10px 0 14px"><span class="status {css}">{label.upper()}</span></div>
      <div class="stat-label">Target value</div><div class="stat-number">{setpoint:.2f}</div>
      <div class="stat-label" style="margin-top:15px">Predicted final output</div><div class="stat-number">{predicted_output:.3f}</div>
      <div class="stat-label" style="margin-top:15px">Predicted signed error</div><div class="stat-number">{predicted_error:+.3f}</div>
      <p class="small-muted" style="margin-top:14px">Signed error = target − predicted output. Positive values indicate the model expects the output to settle below the target.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
      <div class="section-kicker">Control interpretation</div>
      <p class="small-muted">Integral action accumulates error over time and can remove steady-state error in a stable, unsaturated loop. Bias may remain when limits, disturbances, model mismatch or controller constraints intervene.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
c_left, c_right = st.columns([1.1, 1], gap="large")
with c_left:
    st.markdown('<div class="section-kicker">03 / Control effort</div><div class="section-title">Controller output</div>', unsafe_allow_html=True)
    ctrl = go.Figure()
    ctrl.add_trace(go.Scatter(x=response["time"], y=response["control"], mode="lines",
        name="Control effort", line=dict(color="#a78bfa", width=2.2),
        fill="tozeroy", fillcolor="rgba(167,139,250,.08)"))
    ctrl.update_layout(height=270, margin=dict(l=8,r=8,t=8,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,12,25,.32)",
        font=dict(family="Manrope",color="#dce7ff",size=10),
        xaxis=dict(title="Time (s)",gridcolor="rgba(171,192,255,.10)"),
        yaxis=dict(title="Control effort",gridcolor="rgba(171,192,255,.10)"),
        showlegend=False)
    st.plotly_chart(ctrl, use_container_width=True, config={"displayModeBar":False})
with c_right:
    st.markdown('<div class="section-kicker">04 / Feature inspection</div><div class="section-title">Early-window model inputs</div>', unsafe_allow_html=True)
    feature_labels = {
        "initial_error":"Initial error", "early_error":"Error at cutoff",
        "error_integral_early":"Accumulated early error", "mean_abs_error_early":"Mean absolute error",
        "control_mean_early":"Mean control effort", "control_std_early":"Control-effort variation",
        "kp":"Proportional gain (Kp)", "ki":"Integral gain (Ki)",
        "plant_gain":"Plant gain", "time_constant":"Time constant",
    }
    feature_rows = []
    for key, label_text in feature_labels.items():
        value = features[key]
        feature_rows.append(
            f"<div style='display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(171,192,255,.10);padding:7px 0'><span class='small-muted'>{label_text}</span><span style='font-family:DM Mono,monospace;color:#e3efff'>{value:+.4f}</span></div>"
        )
    feature_card = (
        '<div class="glass"><p class="small-muted">These features are calculated from the first 30% of the response. The model does not receive the true final output as an input.</p>'
        + "".join(feature_rows) + "</div>"
    )
    st.markdown(feature_card, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass">
  <div class="section-kicker">Engineering note / Scope</div>
  <p class="small-muted">This is an educational proof of concept trained on synthetic responses. The reference final error is displayed only to compare prediction with this simulated run; it is not a model feature. Real-world use requires logged plant data, time-aware evaluation, calibration, uncertainty estimates and safety review.</p>
</div>
""", unsafe_allow_html=True)

st.caption("MOTIONSCORE P04 · STEADY-STATE ACCURACY PREDICTOR · SYNTHETIC LAB")
