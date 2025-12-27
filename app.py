import streamlit as st

# ---------- Load predictor once ----------
@st.cache_resource
def load_predictor():
    from inference import predict_default, DEFAULT_THRESHOLD
    return predict_default, DEFAULT_THRESHOLD

predict_default, DEFAULT_THRESHOLD = load_predictor()

# ---------- Page config ----------
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    layout="centered"
)

st.title("Credit Default Risk Predictor")
st.write("Enter customer details to estimate default risk.")

# ---------------- Input Section ----------------
st.header("Customer Information")

age = st.number_input("Age", min_value=18, max_value=100, value=40)

Late3059 = st.number_input(
    "Times 30–59 Days Late",
    min_value=0,
    value=0
)

OpenCredit = st.number_input(
    "Number of Open Credit Lines",
    min_value=0,
    value=5
)

Late90 = st.number_input(
    "Times 90+ Days Late",
    min_value=0,
    value=0
)

PropLines = st.number_input(
    "Real Estate Loans / Lines",
    min_value=0,
    value=1
)

Late6089 = st.number_input(
    "Times 60–89 Days Late",
    min_value=0,
    value=0
)

Deps = st.number_input(
    "Number of Dependents",
    min_value=0,
    value=0
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=8000.0
)

DebtRatio = st.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=2000.0
)

UnsecLines = st.number_input(
    "Revolving Utilization of Unsecured Lines",
    min_value=0.0,
    max_value=1.0,
    value=0.4
)

# ---------------- Threshold ----------------
st.header("Decision Threshold")

threshold = st.slider(
    "Default Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01
)

# ---------------- Prediction ----------------
if st.button("Predict Default Risk"):
    raw_input = {
        "age": age,
        "Late3059": Late3059,
        "OpenCredit": OpenCredit,
        "Late90": Late90,
        "PropLines": PropLines,
        "Late6089": Late6089,
        "Deps": Deps,
        "MonthlyIncome": MonthlyIncome,
        "DebtRatio": DebtRatio,
        "UnsecLines": UnsecLines
    }

    try:
        result = predict_default(raw_input, threshold)

        st.subheader("Prediction Result")
        st.write(f"**Default Probability:** {result['probability'] * 100:.2f}%")
        st.write(f"**Decision:** {result['prediction']}")
        st.caption(f"Threshold used: {result['threshold_used']:.2f}")

    except Exception as e:
        st.error("Prediction failed. Check inputs or model artifacts.")
        st.exception(e)
