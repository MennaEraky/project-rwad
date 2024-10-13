import streamlit as st
import pandas as pd
import pickle
import gdown
from sklearn.ensemble import RandomForestRegressor

# Function to download and load the model using gdown
def load_model_from_drive(file_id):
    output = 'vehicle_price_model.pkl'
    try:
        # Construct the Google Drive download URL
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        
        # Load the model from the file
        with open(output, 'rb') as file:
            model = pickle.load(file)
        
        # Check if the loaded model is a RandomForestRegressor
        if isinstance(model, RandomForestRegressor):
            return model
        else:
            st.error("Loaded model is not a RandomForestRegressor.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

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
        file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
        model = load_model_from_drive(file_id)
        
        if model is not None:
            # Preprocess the user input
            input_data = preprocess_input(kilometres, fuel_consumption, doors, seats)
            
            try:
                # Make the prediction
                prediction = model.predict(input_data)
                
                # Display the result
                st.subheader("Predicted Price:")
                st.write(f"${prediction[0]:,.2f}")
                
                # Visualize the result
                st.subheader("Price Visualization")
                st.bar_chart(pd.DataFrame({'Price': [prediction[0]]}, index=['Vehicle']))
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.error("Failed to load the model.")

if __name__ == "__main__":
    main()
