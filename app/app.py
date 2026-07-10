from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "X_train.csv"


# ======================================================
# LOAD MODEL AND FEATURE COLUMNS
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_columns():
    return pd.read_csv(TRAIN_DATA_PATH, nrows=1).columns.tolist()


model = load_model()
feature_columns = load_feature_columns()


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
)


# ======================================================
# CUSTOM STYLE
# ======================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666666;
            margin-bottom: 1.5rem;
        }

        .result-box {
            padding: 1.2rem;
            border-radius: 12px;
            background-color: #f5f7fa;
            border: 1px solid #e5e7eb;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# HEADER
# ======================================================
st.markdown(
    '<div class="main-title">🛍️ Retail Customer Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Classify customers into high-value or standard-value segments '
    'using profile and purchasing behaviour.'
    '</div>',
    unsafe_allow_html=True,
)


# ======================================================
# INPUT FORM
# ======================================================
left_col, right_col = st.columns(2)

with st.form("prediction_form"):

    with left_col:
        st.subheader("Customer Profile")

        age = st.slider(
            "Age",
            min_value=18,
            max_value=70,
            value=35,
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

        city = st.selectbox(
            "City",
            [
                "Kuala Lumpur",
                "Penang",
                "Johor Bahru",
                "Ipoh",
                "Shah Alam",
                "Melaka",
            ],
        )

        membership = st.selectbox(
            "Membership",
            ["Bronze", "Silver", "Gold", "Platinum"],
        )

        favorite_category = st.selectbox(
            "Favourite Product Category",
            ["Electronics", "Fashion", "Beauty", "Home", "Sports"],
        )

        favorite_payment = st.selectbox(
            "Preferred Payment Method",
            [
                "Credit Card",
                "Debit Card",
                "Online Banking",
                "E-Wallet",
                "Cash on Delivery",
            ],
        )

    with right_col:
        st.subheader("Purchase Behaviour")

        total_orders = st.number_input(
            "Total Orders",
            min_value=1,
            max_value=100,
            value=5,
        )

        average_order_value = st.number_input(
            "Average Order Value (RM)",
            min_value=0.0,
            value=400.0,
            step=10.0,
        )

        total_quantity = st.number_input(
            "Total Quantity Purchased",
            min_value=1,
            max_value=500,
            value=10,
        )

        customer_tenure_days = st.number_input(
            "Customer Tenure (Days)",
            min_value=1,
            max_value=2000,
            value=365,
        )

        days_since_last_purchase = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            max_value=1000,
            value=30,
        )

        purchase_frequency = st.number_input(
            "Monthly Purchase Frequency",
            min_value=0.0,
            value=0.5,
            step=0.1,
        )

        average_items_per_order = st.number_input(
            "Average Items per Order",
            min_value=1.0,
            value=2.0,
            step=0.1,
        )

    submitted = st.form_submit_button(
        "Classify Customer",
        use_container_width=True,
    )


# ======================================================
# PREDICTION
# ======================================================
if submitted:
    total_revenue = total_orders * average_order_value

    raw_input = pd.DataFrame(
        [{
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "average_order_value": average_order_value,
            "total_quantity": total_quantity,
            "age": age,
            "gender": gender,
            "city": city,
            "membership": membership,
            "favorite_payment": favorite_payment,
            "favorite_category": favorite_category,
            "customer_tenure_days": customer_tenure_days,
            "days_since_last_purchase": days_since_last_purchase,
            "purchase_frequency": purchase_frequency,
            "average_items_per_order": average_items_per_order,
        }]
    )

    encoded_input = pd.get_dummies(
        raw_input,
        drop_first=True,
    )

    encoded_input = encoded_input.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    prediction = int(model.predict(encoded_input)[0])
    probability = float(
        model.predict_proba(encoded_input)[0, 1]
    )

    st.divider()
    st.subheader("Prediction Result")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Estimated Revenue",
            f"RM{total_revenue:,.2f}",
        )

    with metric_col2:
        st.metric(
            "High-Value Probability",
            f"{probability:.1%}",
        )

    with metric_col3:
        st.metric(
            "Predicted Segment",
            "High Value" if prediction == 1 else "Standard Value",
        )

    if prediction == 1:
        st.success(
            "This customer is classified as a High-Value Customer."
        )
    else:
        st.info(
            "This customer is classified as a Standard-Value Customer."
        )

    st.progress(probability)

    st.caption(
        "This model classifies current customer value based on the "
        "available customer profile and purchasing behaviour."
    )