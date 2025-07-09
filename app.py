import streamlit as st
import numpy as np
import pickle

# Loading model, scaler, and feature names
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
dropdown_options = pickle.load(open('dropdown_options.pkl', 'rb'))  

st.set_page_config(page_title="House Price Predictor")
st.title("House Price Prediction")
st.markdown("Enter property details below:")

dropdown_map = {
    'Locality': 'Locality_',
    'Furnishing': 'Furnishing_',
    'Status': 'Status_',
    'Transaction': 'Transaction_',
    'Type': 'Type_'
}

# Initializing dropdown selections
dropdown_selections = {}

for col, prefix in dropdown_map.items():
    if col in dropdown_options:
        selection = st.selectbox(col, dropdown_options[col])
        dropdown_selections[col] = selection  

# Initializing input vector
user_input = []

for feature in feature_names:
    handled = False

    # Handling one-hot encoded categorical features
    for col, prefix in dropdown_map.items():
        if feature.startswith(prefix):
            selected_value = dropdown_selections.get(col)
            if feature == f"{prefix}{selected_value}":
                user_input.append(1)
            else:
                user_input.append(0)
            handled = True
            break

    if handled:
        continue

    # Handling numerical features with appropriate sliders
    if "Area" in feature:
        val = st.slider("Area (sqft)", 200, 10000, 1000)
    elif "BHK" in feature:
        val = st.slider("Bedrooms (BHK)", 1, 10, 3)
    elif "Bathroom" in feature:
        val = st.slider("Bathrooms", 1, 5, 2)
    elif "Parking" in feature:
        val = st.slider("Parking Spaces", 0, 3, 1)
    elif "Per_Sqft" in feature:
        val = st.slider("Price per Sqft", 1000, 20000, 6000)
    else:
        val = st.slider(f"{feature}", 0.0, 1.0, 0.0)  # fallback

    user_input.append(val)

# Conversion to NumPy array and scale
input_array = np.array([user_input])
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated House Price: ₹ {prediction:.2f}")
