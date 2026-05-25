import math
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Titanic Survival Prediction System",
    page_icon="🚢",
    layout="wide"
)

LEARNING_RATE = 0.1

w_input_hidden = {
    "x1_h1": 0.11,
    "x2_h1": 0.14,
    "x3_h1": 0.17,
    "x1_h2": 0.21,
    "x2_h2": 0.24,
    "x3_h2": 0.27,
}

b_h1 = 0.1
b_h2 = 0.1

w_hidden_output = {
    "h1_o1": 0.31,
    "h2_o1": 0.34,
}

b_o = 0.1

PCLASS_MIN, PCLASS_MAX = 1, 3
AGE_MIN, AGE_MAX = 0, 100
FARE_MIN, FARE_MAX = 0, 150

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def min_max_scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    scaled = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, scaled))

def forward_pass(x1, x2, x3):
    z_h1 = (x1 * w_input_hidden["x1_h1"] +
            x2 * w_input_hidden["x2_h1"] +
            x3 * w_input_hidden["x3_h1"] + b_h1)

    z_h2 = (x1 * w_input_hidden["x1_h2"] +
            x2 * w_input_hidden["x2_h2"] +
            x3 * w_input_hidden["x3_h2"] + b_h2)

    h1 = sigmoid(z_h1)
    h2 = sigmoid(z_h2)

    z_o1 = (h1 * w_hidden_output["h1_o1"] +
            h2 * w_hidden_output["h2_o1"] + b_o)

    y = sigmoid(z_o1)

    return {
        "z_h1": z_h1,
        "z_h2": z_h2,
        "h1": h1,
        "h2": h2,
        "z_o1": z_o1,
        "y": y
    }

def backward_pass(x1, x2, x3, target, fp):
    y = fp["y"]
    h1 = fp["h1"]
    h2 = fp["h2"]

    mse = 0.5 * ((target - y) ** 2)

    delta_o = (target - y) * y * (1 - y)
    delta_h1 = h1 * (1 - h1) * (w_hidden_output["h1_o1"] * delta_o)
    delta_h2 = h2 * (1 - h2) * (w_hidden_output["h2_o1"] * delta_o)

    updated_hidden_output = {
        "h1_o1": w_hidden_output["h1_o1"] + LEARNING_RATE * delta_o * h1,
        "h2_o1": w_hidden_output["h2_o1"] + LEARNING_RATE * delta_o * h2,
    }

    updated_input_hidden = {
        "x1_h1": w_input_hidden["x1_h1"] + LEARNING_RATE * delta_h1 * x1,
        "x2_h1": w_input_hidden["x2_h1"] + LEARNING_RATE * delta_h1 * x2,
        "x3_h1": w_input_hidden["x3_h1"] + LEARNING_RATE * delta_h1 * x3,
        "x1_h2": w_input_hidden["x1_h2"] + LEARNING_RATE * delta_h2 * x1,
        "x2_h2": w_input_hidden["x2_h2"] + LEARNING_RATE * delta_h2 * x2,
        "x3_h2": w_input_hidden["x3_h2"] + LEARNING_RATE * delta_h2 * x3,
    }

    updated_biases = {
        "b_h1": b_h1 + LEARNING_RATE * delta_h1,
        "b_h2": b_h2 + LEARNING_RATE * delta_h2,
        "b_o": b_o + LEARNING_RATE * delta_o,
    }

    return {
        "mse": mse,
        "delta_o": delta_o,
        "delta_h1": delta_h1,
        "delta_h2": delta_h2,
        "updated_hidden_output": updated_hidden_output,
        "updated_input_hidden": updated_input_hidden,
        "updated_biases": updated_biases
    }

def probability_chart(survival_prob):
    non_survival_prob = 1 - survival_prob

    fig = go.Figure(data=[
        go.Bar(
            x=["Not Survived", "Survived"],
            y=[non_survival_prob, survival_prob],
            marker_color=["#ff6b6b", "#2ecc71"],
            text=[f"{non_survival_prob:.2%}", f"{survival_prob:.2%}"],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Prediction Probability",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=400
    )
    return fig

st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0e4c92;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #5c677d;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fbff;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #dbe7f3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .success-box {
        background-color: #eafaf1;
        padding: 1rem;
        border-radius: 12px;
        border-left: 6px solid #2ecc71;
        color: #1e5631;
        font-weight: 600;
    }
    .danger-box {
        background-color: #fff1f0;
        padding: 1rem;
        border-radius: 12px;
        border-left: 6px solid #e74c3c;
        color: #7f1d1d;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<div class="main-title">Titanic Survival Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep Learning Based Passenger Survival Prediction</div>', unsafe_allow_html=True)
with col2:
    st.markdown("## 🚢")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Project Description")
    st.write(
        "This application predicts whether a Titanic passenger would survive based on "
        "Passenger Class, Age, and Fare using an Artificial Neural Network (ANN). "
        "It demonstrates input preprocessing, forward propagation, error calculation, "
        "backpropagation, and weight updates for passenger survival prediction."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Passenger Input Form")

    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
    age = st.slider("Age", min_value=0, max_value=100, value=24)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=120.0, step=1.0)

    predict_btn = st.button("Predict Survival", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Normalization Preview")

    x1 = min_max_scale(pclass, PCLASS_MIN, PCLASS_MAX)
    x2 = min_max_scale(age, AGE_MIN, AGE_MAX)
    x3 = min_max_scale(fare, FARE_MIN, FARE_MAX)

    c1, c2, c3 = st.columns(3)
    c1.metric("Normalized Pclass", f"{x1:.4f}", border=True)
    c2.metric("Normalized Age", f"{x2:.4f}", border=True)
    c3.metric("Normalized Fare", f"{x3:.4f}", border=True)

    st.info("These normalized values are used as inputs to the neural network.")
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    fp = forward_pass(x1, x2, x3)

    survival_prob = fp["y"]
    non_survival_prob = 1 - survival_prob
    prediction = "Survived" if survival_prob > 0.5 else "Not Survived"
    confidence = max(survival_prob, non_survival_prob)

    bp = backward_pass(x1, x2, x3, 1, fp)

    st.write("")
    res1, res2, res3 = st.columns(3)
    res1.metric("Prediction Result", prediction, border=True)
    res2.metric("Survival Probability", f"{survival_prob:.2%}", border=True)
    res3.metric("Confidence Score", f"{confidence:.2%}", border=True)

    if prediction == "Survived":
        st.markdown(f'<div class="success-box">Prediction: {prediction} ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="danger-box">Prediction: {prediction} ❌</div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1, 1])

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Forward Propagation Results")
        st.write(f"Net input at h1: **{fp['z_h1']:.6f}**")
        st.write(f"Net input at h2: **{fp['z_h2']:.6f}**")
        st.write(f"Output of h1: **{fp['h1']:.6f}**")
        st.write(f"Output of h2: **{fp['h2']:.6f}**")
        st.write(f"Net input at output neuron: **{fp['z_o1']:.6f}**")
        st.write(f"Final predicted output: **{fp['y']:.6f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Backpropagation Results")
        st.write(f"Mean Squared Error: **{bp['mse']:.6f}**")
        st.write(f"Output layer gradient (δo): **{bp['delta_o']:.6f}**")
        st.write(f"Hidden layer gradient (δh1): **{bp['delta_h1']:.6f}**")
        st.write(f"Hidden layer gradient (δh2): **{bp['delta_h2']:.6f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    w1, w2 = st.columns(2)

    with w1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Updated Hidden → Output Weights")
        st.write(f"h1 → o1: **{bp['updated_hidden_output']['h1_o1']:.6f}**")
        st.write(f"h2 → o1: **{bp['updated_hidden_output']['h2_o1']:.6f}**")
        st.write(f"Updated output bias bo: **{bp['updated_biases']['b_o']:.6f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with w2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Updated Input → Hidden Weights")
        st.write(f"x1 → h1: **{bp['updated_input_hidden']['x1_h1']:.6f}**")
        st.write(f"x2 → h1: **{bp['updated_input_hidden']['x2_h1']:.6f}**")
        st.write(f"x3 → h1: **{bp['updated_input_hidden']['x3_h1']:.6f}**")
        st.write(f"x1 → h2: **{bp['updated_input_hidden']['x1_h2']:.6f}**")
        st.write(f"x2 → h2: **{bp['updated_input_hidden']['x2_h2']:.6f}**")
        st.write(f"x3 → h2: **{bp['updated_input_hidden']['x3_h2']:.6f}**")
        st.write(f"Updated hidden bias bh1: **{bp['updated_biases']['b_h1']:.6f}**")
        st.write(f"Updated hidden bias bh2: **{bp['updated_biases']['b_h2']:.6f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Visualization Area")
    fig = probability_chart(survival_prob)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)