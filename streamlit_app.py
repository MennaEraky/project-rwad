import streamlit as st
import pandas as pd
import pickle
import gdown
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
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
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    model_features = model.feature_names_in_  # Get the features used during training
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)  # Fill missing columns with 0
    return input_df_encoded

# Create a function to generate plots
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

    # Pie Chart for Fuel Type Distribution
    fuel_distribution = df['FuelType'].value_counts().reset_index()
    fuel_distribution.columns = ['Fuel Type', 'Count']
    pie_chart = px.pie(fuel_distribution, names='Fuel Type', values='Count', 
                       title='Fuel Type Distribution')

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    heatmap_fig = plt.gcf()
    plt.close()

    # Dashboard Layout using Plotly
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Fuel Consumption vs Price', 
                                                         'Price Distribution', 
                                                         'Price by Transmission',
                                                         'Fuel Type Distribution', 
                                                         'Feature Correlation Heatmap'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}],
                               [{"type": "box"}, {"type": "pie"}],
                               [{"colspan": 2, "type": "scatter"}, None]])

    # Adding traces to the subplots
    fig.add_trace(go.Scatter(x=df['FuelConsumption'], y=df['Price'], mode='markers',
                             marker=dict(color=df['FuelType'].apply(lambda x: 'blue' if x == 'Petrol' else 'red')), name='Fuel vs Price'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Price'], nbinsx=30, name='Price Distribution'), row=1, col=2)
    fig.add_trace(go.Box(y=df['Price'], x=df['Transmission'], name='Price by Transmission'), row=2, col=1)
    fig.add_trace(go.Pie(labels=fuel_distribution['Fuel Type'], values=fuel_distribution['Count'], name='Fuel Type Distribution'), row=2, col=2)
    
    # Adding heatmap as a scatter plot with fixed size
    fig.add_trace(go.Scatter(x=corr.index, y=corr.columns, mode='markers', marker=dict(size=8, color='lightblue'), name='Correlation Heatmap', text=corr.values.flatten()), row=3, col=1)

    # Update layout for interactivity and aesthetics
    fig.update_layout(height=800, width=1200, title_text="Vehicle Prices Dashboard", showlegend=False)

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
            }).sort_values('importance', ascending=False).head(10)

            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Top 10 Important Features for Price Prediction')
            st.plotly_chart(fig)

            # Generate and display dashboard
            st.subheader("Vehicle Prices Dashboard")
            dashboard_fig = create_dashboard(input_df_display)
            st.plotly_chart(dashboard_fig)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Run the main function
if __name__ == "__main__":
    main()
