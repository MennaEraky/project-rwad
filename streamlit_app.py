import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    input_df = pd.DataFrame(data, index=[0])  # Create DataFrame with an index
    # One-Hot Encoding for categorical features based on the training model's features
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to ensure it matches the model's expected input
    model_features = model.feature_names_in_  # Get the features used during training
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)  # Fill missing columns with 0
    return input_df_encoded

# Create a function to clean the uploaded DataFrame
def clean_data(df):
    # Replace specific values with NaN
    df.replace(['POA', '-', '- / -'], np.nan, inplace=True)

    # Convert relevant columns to numeric types
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')
    df['FuelConsumption'] = df['FuelConsumption'].str.extract(r'(\d+\.\d+)').astype(float)
    df['Doors'] = df['Doors'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Seats'] = df['Seats'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['CylindersinEngine'] = df['CylindersinEngine'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Engine'] = df['Engine'].str.extract(r'(\d+)').fillna(0).astype(int)

    return df

# Create a function to generate plots including correlation analysis
def create_dashboard(df):
    # Scatter plot for Fuel Consumption vs. Price
    scatter = px.scatter(df, x='FuelConsumption', y='Price', color='FuelType',
                         title='Fuel Consumption vs Price', 
                         labels={'FuelConsumption': 'Fuel Consumption (L/100km)', 'Price': 'Price ($)'})

    # Histogram for Price Distribution
    histogram = px.histogram(df, x='Price', nbins=30, 
                             title='Distribution of Vehicle Prices', 
                             labels={'Price': 'Price ($)'})

    # Box Plot for Price by Transmission Type
    box = px.box(df, x='Transmission', y='Price', 
                 title='Price Distribution by Transmission Type', 
                 labels={'Transmission': 'Transmission Type', 'Price': 'Price ($)'})

    # Create a correlation matrix
    correlation_matrix = df[['Engine', 'CylindersinEngine', 'FuelConsumption', 'Price']].corr()

    # Heatmap for Correlation Matrix
    heatmap_fig = px.imshow(correlation_matrix, text_auto=True,
                             title='Correlation Matrix',
                             color_continuous_scale='Viridis',
                             labels=dict(x="Features", y="Features", color="Correlation Coefficient"))

    # Dashboard Layout using Plotly
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Fuel Consumption vs Price', 'Price Distribution', 'Price by Transmission', 'Correlation Heatmap'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}], [{"type": "box"}, {"type": "heatmap"}]])

    # Adding traces to the subplots
    fig.add_trace(go.Scatter(x=df['FuelConsumption'], y=df['Price'], mode='markers',
                             marker=dict(color=df['FuelType'].apply(lambda x: 'blue' if x == 'Petrol' else 'red')), name='Fuel vs Price'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Price'], nbinsx=30, name='Price Distribution'), row=1, col=2)
    fig.add_trace(go.Box(y=df['Price'], x=df['Transmission'], name='Price by Transmission'), row=2, col=1)

    # Add heatmap
    fig.add_trace(go.Heatmap(z=correlation_matrix.values,
                             x=correlation_matrix.columns,
                             y=correlation_matrix.index,
                             colorscale='Viridis', colorbar=dict(title='Correlation Coefficient')), row=2, col=2)

    # Update layout for interactivity and aesthetics
    fig.update_layout(height=900, width=1200, title_text="Vehicle Prices Dashboard", showlegend=False)

    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="wide")
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

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
        kilometres = st.number_input("Kilometres 🛣", min_value=0, value=50000, step=1000, key="kilometres")
        cylinders_in_engine = st.number_input("Cylinders in Engine 🔢", min_value=1, value=4, key="cylinders_in_engine")
        body_type = st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"], key="body_type")
        doors = st.selectbox("Number of Doors 🚪", [2, 3, 4, 5], key="doors")

    # Load model only once and store in session state
    if 'model' not in st.session_state:
        model_file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID for model
        st.session_state.model = load_model_from_drive(model_file_id)

    # Make prediction automatically based on inputs
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
            input_df_display = pd.DataFrame(input_data, index=[0])
            st.dataframe(input_df_display)

            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': st.session_state.model.feature_names_in_,
                'importance': st.session_state.model.feature_importances_
            }).sort_values(by='importance', ascending=False)

            fig_importance = px.bar(feature_importance, x='feature', y='importance',
                                     title='Feature Importance', labels={'importance': 'Importance', 'feature': 'Feature'},
                                     color='importance', color_continuous_scale='Blues')

            st.plotly_chart(fig_importance)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # File uploader for data visualization
    st.subheader("Upload a CSV File for Correlation Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df_cleaned = clean_data(df)
        
        # Show cleaned data and create visualizations
        st.write("Cleaned Data Preview:")
        st.dataframe(df_cleaned)

        # Create and display dashboard plots
        dashboard_fig = create_dashboard(df_cleaned)
        st.plotly_chart(dashboard_fig)

    st.warning("Please ensure that the uploaded CSV file contains the following columns: 'Price', 'FuelConsumption', 'Kilometres', 'CylindersinEngine', 'Engine', 'Transmission'.")

if __name__ == "__main__":
    main()
