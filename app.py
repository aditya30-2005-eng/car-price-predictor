import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("car_price_model.pkl", "rb"))

# Load dataset (for dropdown values)
df = pd.read_csv("quikr_car.csv")
df = df.dropna()
df["company"] = df["company"].str.strip()
df["name"] = df["name"].str.strip()

companies = sorted(df["company"].unique())
models = sorted(df["name"].unique())

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            text-align: center;
            color: #1a73e8;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #1558b0;
            color: white;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1a73e8;
            color: white;
            text-align: center;
            padding: 10px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .header {
            background-color: #1a73e8;
            color: white;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
    <div class="header">
        <h1>🚗 Car Price Predictor</h1>
        <p>Powered by Machine Learning | Built with ❤️ by <b>Aditya Singh</b></p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### Predict the resale value of your car with AI 🔮")
st.markdown("---")

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("📝 Enter Car Details")

company = st.selectbox("🏢 Select Company", companies)
car_model = st.selectbox("🚘 Select Car Model", [m for m in models if company in m])
category = st.selectbox("📂 Car Category", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Wagon"])
year = st.number_input("📅 Year of Car", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("📍 Kilometers Driven", min_value=0, step=1000)
fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

st.markdown("---")

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict Price"):
    input_data = pd.DataFrame([{
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
        "company": company
    }])

    price = model.predict(input_data)[0]

    st.success(f"💰 Predicted Price: **₹ {int(price):,}**")

    st.info(f"📌 Car: **{company} {car_model}** | Category: **{category}** | Year: **{year}**")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
    <div class="footer">
        <p>© 2025 Car Price Predictor | Developed by <b>Aditya Singh</b></p>
    </div>
""", unsafe_allow_html=True)
