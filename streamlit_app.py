import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # or your preferred library for loading models

# Function to load the model from Google Drive (or other source)
def load_model_from_drive(file_id):
    # Load your model using joblib or any other method
    model = joblib.load(f'https://drive.google.com/uc?id={file_id}')
    return model

# Function to preprocess input data
def preprocess_input(input_data, model):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=['UsedOrNew', 'Transmission', 'DriveType', 'FuelType', 'BodyType'], drop_first=True)
    
    # Ensure all necessary columns for the model are present
    model_columns = model.feature_names_in_  # Get feature names from the model
    missing_cols = set(model_columns) - set(input_df_encoded.columns)

    # Add missing columns with 0 values
    for col in missing_cols:
        input_df_encoded[col] = 0
    
    # Reorder columns to match model's input
    input_df_encoded = input_df_encoded[model_columns]
    
    return input_df_encoded  # Modify this based on your preprocessing needs

# Function to create the main dashboard visualization
def create_dashboard(df):
    # Example: create a scatter plot of price vs. kilometres
    fig = px.scatter(df, x='Kilometres', y='Price', color='FuelType',
                     title='Price vs. Kilometres Driven',
                     labels={'Kilometres': 'Kilometres Driven', 'Price': 'Price'})
    return fig

# Function to generate additional visualizations
def create_additional_visualizations(df, results):
    # Correlation heatmap
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)  # Display heatmap in Streamlit

    # Box plots for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_cols:
        fig = px.box(df, y=col, title=f'Box Plot of {col}')
        st.plotly_chart(fig)

    # Comparison of regression algorithms
    model_names = list(results.keys())
    average_scores = [np.mean(scores) for scores in results.values()]

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, average_scores, color='skyblue')
    plt.xlabel('Average Cross-Validation Score')
    plt.title('Comparison of Regression Algorithms')
    plt.gca().invert_yaxis()
    st.pyplot(plt)  # Display bar chart in Streamlit

# Main function to run the Streamlit app
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

            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': st.session_state.model.feature_names_in_,
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            # Plotting feature importance using plotly
            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Top 10 Important Features', labels={'importance': 'Importance', 'feature': 'Feature'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig)

            # Displaying input data and prediction as a table
            st.subheader("Input Data and Prediction")
            input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
            input_df_display = pd.DataFrame(input_data, index=[0])
            st.dataframe(input_df_display)

            # Data Upload Section
            st.markdown("---")
            st.header("📊 Upload Your Vehicle Data for Visualization")

            # File uploader
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("Data loaded successfully!")

                    # Create and display the dashboard
                    st.subheader("Vehicle Prices Dashboard")
                    dashboard_fig = create_dashboard(df)
                    st.plotly_chart(dashboard_fig)

                    # Create additional visualizations
                    st.markdown("---")
                    st.subheader("Additional Visualizations")

                    # Results should be a dictionary of models and their cross-validation scores
                    results = {
                        # Example model results; replace these with actual results
                        'Model A': [0.8, 0.85, 0.82],
                        'Model B': [0.75, 0.78, 0.76],
                        'Model C': [0.88, 0.86, 0.87]
                    }
                    create_additional_visualizations(df, results)

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Failed to load the model.")

if __name__ == "__main__":
    main()
