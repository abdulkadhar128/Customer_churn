import streamlit as st
import joblib
import numpy as np

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("customer_churn_model.pkl")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details and click Predict.")

# ==========================================
# INPUTS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=60,
        value=12,
        step=1
    )

    usage_frequency = st.number_input(
        "Usage Frequency",
        min_value=0,
        max_value=30,
        value=10,
        step=1
    )

    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

    total_spend = st.number_input(
        "Total Spend",
        min_value=0,
        max_value=1000,
        value=500,
        step=1
    )

    last_interaction = st.number_input(
        "Last Interaction",
        min_value=0,
        max_value=30,
        value=10,
        step=1
    )

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Standard"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

# ==========================================
# ENCODING
# ==========================================

gender = 1 if gender == "Male" else 0

subscription_map = {
    "Basic": 0,
    "Premium": 1,
    "Standard": 2
}

subscription_type = subscription_map[subscription_type]

contract_map = {
    "Annual": 0,
    "Monthly": 1,
    "Quarterly": 2
}

contract_length = contract_map[contract_length]

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("🔍 Predict"):

    features = np.array([[
        age,
        gender,
        tenure,
        usage_frequency,
        support_calls,
        payment_delay,
        subscription_type,
        contract_length,
        total_spend,
        last_interaction
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.subheader("Prediction Confidence")

    st.metric(
        "Churn Risk",
        f"{probability[1] * 100:.2f}%"
    )

    st.metric(
        "Stay Probability",
        f"{probability[0] * 100:.2f}%"
    )

    st.subheader("Customer Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {'Male' if gender == 1 else 'Female'}")
    st.write(f"**Tenure:** {tenure}")
    st.write(f"**Usage Frequency:** {usage_frequency}")
    st.write(f"**Support Calls:** {support_calls}")
    st.write(f"**Payment Delay:** {payment_delay}")
    st.write(f"**Subscription Type:** {['Basic','Premium','Standard'][subscription_type]}")
    st.write(f"**Total Spend:** {total_spend}")
    st.write(f"**Last Interaction:** {last_interaction}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.caption("Customer Churn Prediction using XGBoost and Streamlit")