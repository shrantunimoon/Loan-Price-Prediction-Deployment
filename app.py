
# PREMIUM BANK PORTAL  STREAMLIT APP


import streamlit as st
import pandas as pd
import pickle

# 1️ App Config

st.set_page_config(
    page_title="Loan Prediction System",
    layout="wide",
    page_icon="🏦"
)


# 2️ THEME SETTINGS (Edit only this section to change UI)

PRIMARY_GRADIENT = "linear-gradient(90deg, #6A82FB, #FC5C7D)"
PRIMARY_COLOR = "#6A82FB"
SECONDARY_COLOR = "#FC5C7D"
TEXT_DARK = "#2C2C2C"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F4F7FB"
FONT = "'Segoe UI', sans-serif"

APP_TITLE = "🏦 Premium Loan Status Predictor"
FOOTER_TEXT = "Developed with ❤️ by Deepraj | Powered by Streamlit"

# 3️ Inject Premium CSS

custom_css = f"""
<style>

body {{
    background: {PAGE_BG};
    font-family: {FONT};
}}

.title-box {{
    background: {PRIMARY_GRADIENT};
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: white;
    letter-spacing: 1px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.18);
}}

.card {{
    background: {CARD_BG};
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 8px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}}

.section-header {{
    font-size: 22px;
    font-weight: 600;
    color: {PRIMARY_COLOR};
    padding-bottom: 6px;
    border-bottom: 2px solid {PRIMARY_COLOR};
    margin-bottom: 12px;
}}

.predict-btn > button {{
    background: {PRIMARY_GRADIENT};
    width: 100%;
    font-size: 22px;
    font-weight: 600;
    padding: 14px 20px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
    color: white;
}}

.predict-btn > button:hover {{
    background: {SECONDARY_COLOR};
    transform: scale(1.03);
}}

.result-approved {{
    background: #d1ffe0;
    border-left: 8px solid #16C47F;
    padding: 18px;
    border-radius: 12px;
    font-size: 24px;
    color: #0B4635;
    text-align: center;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
}}

.result-rejected {{
    background: #ffe1e4;
    border-left: 8px solid #D63230;
    padding: 18px;
    border-radius: 12px;
    font-size: 24px;
    color: #5A1E1B;
    text-align: center;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
}}

footer {{
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-top: 50px;
}}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# 4️ Load ML Artifacts

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
cat_cols = pickle.load(open("cat_cols.pkl", "rb"))
num_cols = pickle.load(open("num_cols.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))

# 5️ Header

st.markdown(f"<div class='title-box'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.write("### Fill customer details below to predict loan status.")


# 6️ Input Section


col1, col2 = st.columns(2)
user_inputs = {}

# --------- Categorical Inputs ----------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Categorical Details</div>", unsafe_allow_html=True)

    for col in cat_cols:
        user_inputs[col] = st.selectbox(
            col.replace("_", " ").title(),
            label_encoders[col].classes_
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------- Numerical Inputs ----------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Numerical Details</div>", unsafe_allow_html=True)

    for col in num_cols:
        user_inputs[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=0.0,
            step=1.0
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Convert input → DataFrame
df_input = pd.DataFrame([user_inputs])


# 7️ Pre-processing

for col in cat_cols:
    df_input[col] = label_encoders[col].transform(df_input[col])

df_input[num_cols] = scaler.transform(df_input[num_cols])
df_input = df_input.reindex(columns=feature_order)

# 8️ Prediction Button


st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status", key="predict", help="Click to check eligibility"):
    prediction = model.predict(df_input)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == "Approved":
        st.markdown("<div class='result-approved'> Loan Approved!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-rejected'> Loan Rejected!</div>", unsafe_allow_html=True)


