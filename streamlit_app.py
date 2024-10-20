import streamlit as st
import pandas as pd
import pickle
import gdown
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
    input_df = pd.DataFrame(data, index=[0])
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    model_features = model.feature_names_in_
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)
    return input_df_encoded

# Function to preprocess uploaded dataset
def preprocess_uploaded_data(df, model):
    df_encoded = pd.get_dummies(df, columns=['UsedOrNew', 'Transmission', 'DriveType', 'FuelType', 'BodyType'], drop_first=True)
    model_columns = model.feature_names_in_
    missing_cols = set(model_columns) - set(df_encoded.columns)

    for col in missing_cols:
        df_encoded[col] = 0

    df_encoded = df_encoded[model_columns]
    
    return df_encoded

# Create a function to generate plots
def create_dashboard(df):
    scatter = px.scatter(df, x='FuelConsumption', y='Price', color='FuelType',
                         title='Fuel Consumption vs Price', 
                         labels={'FuelConsumption': 'Fuel Consumption (L/100km)', 'Price': 'Price ($)'})

    histogram = px.histogram(df, x='Price', nbins=30, 
                             title='Distribution of Vehicle Prices', 
                             labels={'Price': 'Price ($)'})

    box = px.box(df, x='Transmission', y='Price', 
                 title='Price Distribution by Transmission Type', 
                 labels={'Transmission': 'Transmission Type', 'Price': 'Price ($)'})

    fig = make_subplots(rows=2, cols=2, subplot_titles=('Fuel Consumption vs Price', 'Price Distribution', 'Price by Transmission'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}], [{"type": "box"}, None]])

    fig.add_trace(go.Scatter(x=df['FuelConsumption'], y=df['Price'], mode='markers',
                             marker=dict(color=df['FuelType'].apply(lambda x: 'blue' if x == 'Petrol' else 'red')), name='Fuel vs Price'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Price'], nbinsx=30, name='Price Distribution'), row=1, col=2)
    fig.add_trace(go.Box(y=df['Price'], x=df['Transmission'], name='Price by Transmission'), row=2, col=1)

    fig.update_layout(height=800, width=1200, title_text="Vehicle Prices Dashboard", showlegend=False)

    return fig

# Function to create additional visualizations
def create_additional_visualizations(df):
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, vmin=-1, vmax=1)
    st.pyplot(plt)

    # Box plots for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        fig = px.box(df, y=col, title=f'Box Plot for {col}')
        st.plotly_chart(fig)

    # Example Model Results for comparison (Replace this with your actual model results)
    results = {
        'Model A': [0.8, 0.85, 0.82],
        'Model B': [0.75, 0.78, 0.76],
        'Model C': [0.88, 0.86, 0.87]
    }
    model_names = list(results.keys())
    average_scores = [np.mean(scores) for scores in results.values()]

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, average_scores, color='skyblue')
    plt.xlabel('Average Cross-Validation Score')
    plt.title('Comparison of Regression Algorithms')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

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

            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': st.session_state.model.feature_names_in_,
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Top 10 Important Features', labels={'importance': 'Importance', 'feature': 'Feature'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig)

            # Displaying input data and prediction as a table
            st.subheader("Input Data and Prediction")
            input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
            st.write(pd.DataFrame([input_data]).T, header=None)

            # Dashboard
            st.subheader("Vehicle Prices Dashboard")
            dashboard_fig = create_dashboard(pd.read_csv('your_vehicle_data.csv'))  # Update with your vehicle data source
            st.plotly_chart(dashboard_fig)

            # Additional visualizations
            st.markdown("---")
            st.subheader("Additional Visualizations")
            create_additional_visualizations(pd.read_csv('your_vehicle_data.csv'))  # Update with your vehicle data source

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Data Upload Section
    st.markdown("---")
    st.header("📊 Upload Your Vehicle Data for Visualization")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")

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
