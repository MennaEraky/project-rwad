import streamlit as st
import pandas as pd
import plotly.express as px
import gdown
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Function to download and load the dataset using gdown
def load_dataset_from_drive(file_id):
    output = 'vehicle_data.csv'
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        return output  # Return the name of the downloaded file
    except Exception as e:
        st.error(f"Error loading the dataset: {str(e)}")
        return None

# Function to create a dashboard visualization
def create_dashboard(data):
    fig = px.scatter(data, x='Year', y='Price', color='FuelType', title='Price vs Year by Fuel Type')
    return fig

def main():
    st.title("Vehicle Price Prediction Dashboard")

    # Load the model
    model = joblib.load('https://drive.google.com/uc?id=11btPBNR74na_NjjnjrrYT8RSf8ffiumo')  # Replace with your model file ID

    # Load dataset only once and store in session state
    if 'dataset' not in st.session_state:
        dataset_file_id = '1BMO9pcLUsx970KDTw1kHNkXg2ghGJVBs'  # Replace with your Google Drive file ID for the dataset
        st.session_state.dataset = load_dataset_from_drive(dataset_file_id)

    # Create the dashboard plots
    st.subheader("Dashboard Visualizations")
    if st.session_state.dataset is not None:
        dashboard_fig = create_dashboard(pd.read_csv(st.session_state.dataset))  # Load your dataset for dashboard visualizations
        st.plotly_chart(dashboard_fig)

    # User input for prediction
    st.subheader("Predict Vehicle Price")
    year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
    used_or_new = st.selectbox("Used or New", ['Used', 'New'])
    transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])
    engine = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=2.0)
    drive_type = st.selectbox("Drive Type", ['FWD', 'RWD', 'AWD'])
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric'])
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=5.0)
    kilometers = st.number_input("Kilometers Driven", min_value=0, value=50000)
    cylinders = st.number_input("Cylinders in Engine", min_value=1, value=4)
    body_type = st.selectbox("Body Type", ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe'])
    doors = st.selectbox("Number of Doors", [2, 3, 4, 5])

    # Predict button
    if st.button("Predict Price"):
        input_data = [[year, used_or_new, transmission, engine, drive_type, fuel_type, fuel_consumption, kilometers, cylinders, body_type, doors]]
        prediction = model.predict(input_data)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
