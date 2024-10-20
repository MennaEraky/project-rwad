import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Function to download and load the model using gdown
def load_model_from_drive(file_id):
    output = 'vehicle_price_model.pkl'
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with open(output, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data, model):
    input_df = pd.DataFrame(data, index=[0])  # Create DataFrame with an index
    # One-Hot Encoding for categorical features based on the training model's features
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to ensure it matches the model's expected input
    model_features = model.feature_names_in_  # Get the features used during training
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)  # Fill missing columns with 0
    return input_df_encoded

# Load the dataset from Google Drive
def load_dataset(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Data cleaning and preprocessing function
def clean_data(df):
    # Replace certain values with NaN
    df.replace(['POA', '-', '- / -'], np.nan, inplace=True)
    
    # Convert relevant columns to numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')
    
    # Extract numeric values from string columns
    df['FuelConsumption'] = df['FuelConsumption'].str.extract(r'(\d+\.\d+)').astype(float)
    df['Doors'] = df['Doors'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Seats'] = df['Seats'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['CylindersinEngine'] = df['CylindersinEngine'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Engine'] = df['Engine'].str.extract(r'(\d+)').fillna(0).astype(int)

    # Fill NaN values for specific columns
    df[['Kilometres', 'FuelConsumption']] = df[['Kilometres', 'FuelConsumption']].fillna(df[['Kilometres', 'FuelConsumption']].median())
    df.dropna(subset=['Year', 'Price'], inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['Brand', 'Model', 'Car/Suv', 'Title', 'Location', 'ColourExtInt', 'Seats'], inplace=True)

    # Label encoding for categorical features
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Create a function to visualize correlations
def visualize_correlations(df):
    # Calculate the correlation matrix
    correlation = df.corr()
    correlation_with_price = correlation['Price']
    
    # Plot correlation
    st.subheader("Correlation with Price")
    st.write(correlation_with_price)

    # Heatmap of the correlation matrix
    fig = px.imshow(correlation, text_auto=True, aspect="auto",
                    title="Correlation Heatmap")
    st.plotly_chart(fig)

# Create additional visualizations
def additional_visualizations(df):
    st.subheader("Price vs Engine Size")
    fig_engine = px.scatter(df, x='Engine', y='Price', title='Price vs Engine Size', 
                             labels={'Engine': 'Engine Size (L)', 'Price': 'Price'},
                             trendline='ols')
    st.plotly_chart(fig_engine)

    st.subheader("Price vs Number of Cylinders")
    fig_cylinders = px.box(df, x='CylindersinEngine', y='Price', 
                            title='Price Distribution by Number of Cylinders',
                            labels={'CylindersinEngine': 'Cylinders in Engine', 'Price': 'Price'})
    st.plotly_chart(fig_cylinders)

    st.subheader("Price vs Fuel Consumption")
    fig_fuel = px.scatter(df, x='FuelConsumption', y='Price', title='Price vs Fuel Consumption',
                          labels={'FuelConsumption': 'Fuel Consumption (L/100 km)', 'Price': 'Price'},
                          trendline='ols')
    st.plotly_chart(fig_fuel)

# Visualize model performance metrics
def visualize_model_performance():
    models = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR",
        "KNeighborsRegressor",
        "MLPRegressor",
        "AdaBoostRegressor",
        "BaggingRegressor",
        "ExtraTreesRegressor"
    ]
    
    scores = [
        [0.38643429, 0.35310009, 0.36801071],
        [0.38620243, 0.35350286, 0.36843282],
        [0.38620616, 0.35349711, 0.36843277],
        [0.33686675, 0.31415677, 0.32787848],
        [0.62213917, 0.40638212, 0.47242902],
        [0.74799343, 0.70412406, 0.70161075],
        [0.73002938, 0.70887856, 0.70533151],
        [-0.03261018, -0.05532926, -0.05188942],
        [0.64170728, 0.63380643, 0.64356449],
        [-0.38015855, -0.41194531, -0.41229902],
        [0.0021934, -0.43429876, -0.28546934],
        [0.72923447, 0.70932019, 0.67318744],
        [0.74919345, 0.70561132, 0.68979889]
    ]
    
    mean_scores = [np.mean(score) for score in scores]
    
    # Create DataFrame for plotting
    performance_df = pd.DataFrame({
        'Model': models,
        'Mean R² Score': mean_scores
    })
    
    # Plot the performance
    st.subheader("Model Performance Comparison")
    fig_performance = px.bar(performance_df, x='Model', y='Mean R² Score', 
                              title='Mean R² Score of Regression Models', 
                              labels={'Mean R² Score': 'Mean R² Score'},
                              color='Mean R² Score', 
                              color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_performance)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="wide")
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is not None:
            df = clean_data(df)

            # Visualizations
            visualize_correlations(df)
            additional_visualizations(df)
            visualize_model_performance()

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020, key="year")
        used_or_new = st.selectbox("Used or New 🏷", ["Used", "New"], key="used_or_new")
        transmission = st.selectbox("Transmission ⚙", ["Manual", "Automatic"], key="transmission")
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1, key="engine")
        drive_type = st.selectbox("Drive Type 🛣", ["FWD", "RWD", "AWD"], key="drive_type")
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel_type")

    with col2:
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1, key="fuel_consumption")
        kilometres = st.number_input("Kilometres Driven 🚗", min_value=0, value=10000, key="kilometres")
        cylinders = st.number_input("Cylinders in Engine 🔥", min_value=0, value=4, key="cylinders")
        doors = st.number_input("Number of Doors 🚪", min_value=1, value=4, key="doors")
        seats = st.number_input("Number of Seats 🪑", min_value=1, value=5, key="seats")

    # Load the model
    model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID for model
    model = load_model_from_drive(model_file_id)
    
    if model is not None:
        # Predict button
        if st.button("Predict Price 💰"):
            try:
                user_input = {
                    'Year': year,
                    'UsedOrNew': 1 if used_or_new == "Used" else 0,
                    'Transmission': transmission,
                    'Engine': engine,
                    'DriveType': drive_type,
                    'FuelType': fuel_type,
                    'FuelConsumption': fuel_consumption,
                    'Kilometres': kilometres,
                    'CylindersinEngine': cylinders,
                    'Doors': doors,
                    'Seats': seats
                }
                input_df_encoded = preprocess_input(user_input, model)
                prediction = model.predict(input_df_encoded)

                st.success(f"The predicted price of the vehicle is: ${prediction[0]:,.2f}")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
