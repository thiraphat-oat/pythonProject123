import joblib
import pandas as pd
import streamlit as st

# Load model and encoders
model = joblib.load('random_forest_regressor_updated.joblib')
label_encoders = joblib.load('label_encoders_car_features.joblib')

# Access each encoder
le_vehicle_type = label_encoders['vehicleType']
le_gearbox = label_encoders['gearbox']
le_fuel_type = label_encoders['fuelType']
le_not_repaired_damage = label_encoders['notRepairedDamage']

# Apply custom CSS for a modern look
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F39C12;
    }
    .stButton button {
        background-color: #F39C12;
        color: white;
        border-radius: 8px;
        border: 2px solid #F39C12;
        font-size: 18px;
        padding: 8px 16px;
        margin: 16px 0;
    }
    .stButton button:hover {
        background-color: white;
        color: #F39C12;
    }
    .stSelectbox, .stSlider {
        font-size: 16px;
    }
    .stSlider>div>div {
        color: #F39C12;
    }
    .stMarkdown {
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description with an icon
st.title('🚗 Car Price Prediction App')
st.markdown("""
    **Use the form below to input your car details and predict the estimated price.**
    \n_Ensure the details are within the specified ranges for accurate predictions._
""")

# Create a form with enhanced styling
with st.form(key='car_price_form'):
    st.header("🚙 Car Details")

    # Input fields with icons and tooltips
    col1, col2 = st.columns(2)
    with col1:
        vehicleType = st.selectbox("🚘 Vehicle Type", le_vehicle_type.classes_)
        yearOfRegistration = st.slider("📅 Year of Registration", min_value=1990, max_value=2016, value=2000, step=1)
        gearbox = st.selectbox("⚙️ Gearbox", le_gearbox.classes_)
        powerPS = st.slider("🏎️ Power (PS)", min_value=50, max_value=250, value=100, step=1)
        kilometer = st.slider("🚗 Kilometer", min_value=0, max_value=150000, value=50000, step=100)
        fuelType = st.selectbox("⛽ Fuel Type", le_fuel_type.classes_)
        notRepairedDamage = st.selectbox("🔧 Not Repaired Damage", le_not_repaired_damage.classes_)

    # Submit button with custom style
    submit_button = st.form_submit_button(label='🔮 Predict Price')

# Prepare data for prediction
if submit_button:
    # Transform the selected values into their encoded forms for model prediction
    input_data = pd.DataFrame({
        'vehicleType': [le_vehicle_type.transform([vehicleType])[0]],
        'yearOfRegistration': [yearOfRegistration],
        'gearbox': [le_gearbox.transform([gearbox])[0]],
        'powerPS': [powerPS],
        'kilometer': [kilometer],
        'fuelType': [le_fuel_type.transform([fuelType])[0]],
        'notRepairedDamage': [le_not_repaired_damage.transform([notRepairedDamage])[0]]
    })

    # Predict
    try:
        prediction = model.predict(input_data)
        predicted_price_baht = prediction[0] * 1_000_000  # คูณด้วย 1,000,000 เพื่อแปลงเป็นบาท
        formatted_price = f"{predicted_price_baht:,.0f} บาท"
        st.success(f"💰 The predicted car price is: **{formatted_price}**.")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
