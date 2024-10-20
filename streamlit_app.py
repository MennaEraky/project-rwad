import streamlit as st
import pandas as pd
import pickle
import gdown
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Function to download and load the model using gdown
def load_model_from_drive(file_id):
    output = 'vehicle_price_model.pkl'
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with open(output, 'rb') as file:
            model = pickle.load(file)
        if isinstance(model, RandomForestRegressor):
            return model
        else:
            st.error("Loaded model is not a RandomForestRegressor.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data, model):
    input_df = pd.DataFrame(data, index=[0])
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    model_features = model.feature_names_in_
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)
    return input_df_encoded

# Function to preprocess uploaded dataset
def preprocess_uploaded_data(df, model):
    # Convert columns with mixed types to numeric if applicable
    for col in ['Engine', 'Kilometres', 'CylindersinEngine', 'FuelConsumption']:
        df[col] = pd.to_numeric(df[col].str.extract('(\d+\.?\d*)')[0], errors='coerce')

    df_encoded = pd.get_dummies(df, columns=['UsedOrNew', 'Transmission', 'DriveType', 'FuelType', 'BodyType'], drop_first=True)
    model_columns = model.feature_names_in_
    missing_cols = set(model_columns) - set(df_encoded.columns)

    for col in missing_cols:
        df_encoded[col] = 0

    df_encoded = df_encoded[model_columns]
    
    return df_encoded

# Main Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="wide")
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020, key="year")
        used_or_new = st.selectbox("Used or New 🏷️", ["Used", "New"], key="used_or_new")
        transmission = st.selectbox("Transmission ⚙️", ["Manual", "Automatic"], key="transmission")
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1, key="engine")
        drive_type = st.selectbox("Drive Type 🛣️", ["FWD", "RWD", "AWD"], key="drive_type")
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel_type")

    with col2:
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1, key="fuel_consumption")
        kilometres = st.number_input("Kilometres 🛣️", min_value=0, value=50000, step=1000, key="kilometres")
        cylinders_in_engine = st.number_input("Cylinders in Engine 🔢", min_value=1, value=4, key="cylinders_in_engine")
        body_type = st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"], key="body_type")
        doors = st.selectbox("Number of Doors 🚪", [2, 3, 4, 5], key="doors")

    if 'model' not in st.session_state:
        model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'
        st.session_state.model = load_model_from_drive(model_file_id)

    if st.session_state.model is not None:
        input_data = {
            'Year': year,
            'UsedOrNew': used_or_new,
            'Transmission': transmission,
            'Engine': engine,
            'DriveType': drive_type,
            'FuelType': fuel_type,
            'FuelConsumption': fuel_consumption,
            'Kilometres': kilometres,
            'CylindersinEngine': cylinders_in_engine,
            'BodyType': body_type,
            'Doors': doors
        }
        input_df = preprocess_input(input_data, st.session_state.model)

        try:
            prediction = st.session_state.model.predict(input_df)

            # Styled prediction display
            st.markdown(f"""
                <div style="font-size: 24px; padding: 10px; background-color: #f0f4f8; border: 2px solid #3e9f7d; border-radius: 5px; text-align: center;">
                    <strong>Predicted Price:</strong> ${prediction[0]:,.2f}
                </div>
            """, unsafe_allow_html=True)

            # Displaying input data and prediction as a table
            st.subheader("Input Data and Prediction")
            input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
            st.write(pd.DataFrame([input_data]).T, header=None)

            # Dashboard and additional visualizations
            st.markdown("---")
            st.header("📊 Upload Your Vehicle Data for Visualization")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("Data loaded successfully!")

                    # Inspect the uploaded data
                    st.write("### Uploaded Data Preview")
                    st.write(df.head())  # Show the first few rows of the uploaded data

                    # Preprocess the uploaded data
                    df_encoded = preprocess_uploaded_data(df, st.session_state.model)

                    # Make predictions on the preprocessed data
                    predictions = st.session_state.model.predict(df_encoded)

                    # Add predictions to the dataframe
                    df['Predicted Price'] = predictions

                    # Create and display the dashboard
                    st.subheader("Vehicle Prices Dashboard")
                    dashboard_fig = create_dashboard(df)
                    st.plotly_chart(dashboard_fig)

                    # Create additional visualizations
                    st.markdown("---")
                    st.subheader("Additional Visualizations")
                    create_additional_visualizations(df)

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
