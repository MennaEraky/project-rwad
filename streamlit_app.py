import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Function to download the model from Google Drive
def download_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?export=download&id={19Y_7fbDCIWD2el7nzH6rVY15DRRcg2oK}'
    response = requests.get(url)
    return response.content

# Load the trained model
def load_model_from_drive(file_id):
    model_data = download_model_from_drive(file_id)
    model = pickle.loads(model_data)
    return model

# Preprocess the input data
def preprocess_input(kilometres, fuel_consumption, doors, seats):
    input_df = pd.DataFrame({
        'Kilometres': [kilometres],
        'FuelConsumption': [fuel_consumption],
        'Doors': [doors],
        'Seats': [seats]
    })
    return input_df

# Main Streamlit app
def main():
    st.title("Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    # User input fields
    kilometres = st.number_input("Kilometres", min_value=0, value=50000)
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)
    doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
    seats = st.selectbox("Number of Seats", [2, 4, 5, 7])
    
    # Button for prediction
    if st.button("Predict Price"):
        # Replace 'your_drive_file_id' with the actual file ID from Google Drive
        model = load_model_from_drive('your_drive_file_id')
        
        # Preprocess the user input
        input_data = preprocess_input(kilometres, fuel_consumption, doors, seats)
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.subheader("Predicted Price:")
        st.write(f"${prediction[0]:,.2f}")
        
        # Visualize the result
        st.subheader("Price Visualization")
        st.bar_chart(pd.DataFrame({'Price': [prediction[0]]}, index=['Vehicle']))

if __name__ == "__main__":
    main()
