import streamlit as st
import pandas as pd
import pickle
import gdown
from sklearn.ensemble import RandomForestRegressor

# Function to download and load the model using gdown
def load_model(file_id):
    model_path = 'vehicle_price_prediction_model.pkl'
    try:
        model_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(model_url, model_path, quiet=False)
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        if isinstance(loaded_model, RandomForestRegressor):
            return loaded_model
        else:
            st.error("The loaded model is not a RandomForestRegressor instance.")
            return None
    except Exception as error:
        st.error(f"An error occurred while loading the model: {str(error)}")
        return None

# Preprocess the input data
def prepare_input_data(input_data, trained_model):
    input_dataframe = pd.DataFrame(input_data, index=[0])  # Create DataFrame with an index
    # One-Hot Encoding for categorical features based on the training model's features
    encoded_input = pd.get_dummies(input_dataframe, drop_first=True)

    # Reindex to match the model's expected input features
    model_columns = trained_model.feature_names_in_  # Get features used during training
    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)  # Fill missing columns with 0
    return encoded_input

# Main Streamlit app
def app():
    st.title("Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    # Create input fields for all required features
    vehicle_year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
    vehicle_condition = st.selectbox("Used or New", ["Used", "New"])
    vehicle_transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)
    vehicle_drive_type = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"])
    vehicle_fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)
    vehicle_kilometres = st.number_input("Kilometres", min_value=0, value=50000)
    engine_cylinders = st.number_input("Cylinders in Engine", min_value=1, value=4)
    vehicle_body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"])
    door_count = st.selectbox("Number of Doors", [2, 3, 4, 5])

    # Button for prediction
    if st.button("Predict Price"):
        google_drive_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
        trained_model = load_model(google_drive_file_id)

        if trained_model is not None:
            # Prepare the user input
            user_input = {
                'Year': vehicle_year,
                'UsedOrNew': vehicle_condition,
                'Transmission': vehicle_transmission,
                'Engine': engine_size,
                'DriveType': vehicle_drive_type,
                'FuelType': vehicle_fuel_type,
                'FuelConsumption': fuel_consumption,
                'Kilometres': vehicle_kilometres,
                'CylindersinEngine': engine_cylinders,
                'BodyType': vehicle_body_type,
                'Doors': door_count
            }
            prepared_input = prepare_input_data(user_input, trained_model)

            try:
                # Make the prediction
                predicted_price = trained_model.predict(prepared_input)

                # Display the result
                st.subheader("Predicted Price:")
                st.write(f"${predicted_price[0]:,.2f}")

                # # Visualize the result
                # st.subheader("Price Visualization")
                # st.bar_chart(pd.DataFrame({'Price': [predicted_price[0]]}, index=['Vehicle']))
            except Exception as error:
                st.error(f"An error occurred while making the prediction: {str(error)}")
        else:
            st.error("Model loading failed.")

if __name__ == "__main__":
    app()
