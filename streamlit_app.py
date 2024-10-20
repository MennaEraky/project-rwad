import streamlit as st
import pandas as pd
import pickle
import gdown
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Function to download and load the model using gdown
@st.cache_resource
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

# Main Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Prediction", layout="wide")
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    # Create input fields for all required features
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
        used_or_new = st.selectbox("Used or New", ["Used", "New"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        engine = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)
        drive_type = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)
        kilometres = st.number_input("Kilometres", min_value=0, value=50000)
        cylinders_in_engine = st.number_input("Cylinders in Engine", min_value=1, value=4)
        body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"])
        doors = st.selectbox("Number of Doors", [2, 3, 4, 5])

    # Load model once
    model = load_model_from_drive('11btPBNR74na_NjjnjrrYT8RSf8ffiumo')

    if model is not None:
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
        input_df = preprocess_input(input_data, model)

        # Prediction Button
        if st.button("Predict Price"):
            try:
                prediction = model.predict(input_df)
                st.subheader("Predicted Price:")
                st.write(f"${prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    # Data Upload Section
    st.markdown("---")
    st.header("📊 Upload Your Vehicle Data for Visualization")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")

            # Visualizations Section
            if st.checkbox("Show Data Visualizations"):
                # Ensure the required columns are in the uploaded dataset
                if all(col in df.columns for col in ['Kilometres', 'Price', 'FuelType', 'Transmission']):
                    fig1 = px.scatter(df, x='Kilometres', y='Price', color='FuelType', title="Kilometres vs Price")
                    st.plotly_chart(fig1, use_container_width=True)

                    fig2 = px.histogram(df, x='Price', nbins=30, title="Price Distribution")
                    st.plotly_chart(fig2, use_container_width=True)

                    fig3 = px.box(df, x='Transmission', y='Price', title="Price by Transmission")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.error("The uploaded file must contain 'Kilometres', 'Price', 'FuelType', and 'Transmission' columns.")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
