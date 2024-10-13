import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Preprocess the input data
def preprocess_input(kilometres, fuel_consumption, doors, seats):
    # Create a DataFrame with the input values
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
        # Load the model
        model = load_model('vehicle_price_model.pkl')  # Replace with your actual model file path
        
        # Preprocess the user input
        input_data = preprocess_input(kilometres, fuel_consumption, doors, seats)
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.subheader("Predicted Price:")
        st.write(f"${prediction[0]:,.2f}")
        
        # Visualize the result (example: bar chart)
        st.subheader("Price Visualization")
        st.bar_chart(pd.DataFrame({'Price': [prediction[0]]}, index=['Vehicle']))

if __name__ == "__main__":
    main()
