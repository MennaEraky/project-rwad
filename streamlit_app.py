import streamlit as st
import pandas as pd
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

    # Dashboard Layout using Plotly
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Fuel Consumption vs Price', 'Price Distribution', 'Price by Transmission'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}], [{"type": "box"}, None]])

    # Adding traces to the subplots with neutral colors
    fig.add_trace(go.Scatter(x=df['FuelConsumption'], y=df['Price'], mode='markers',
                             marker=dict(color='gray'), name='Fuel vs Price'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Price'], nbinsx=30, name='Price Distribution', marker_color='lightgray'), row=1, col=2)
    fig.add_trace(go.Box(y=df['Price'], x=df['Transmission'], name='Price by Transmission', marker_color='lightgray'), row=2, col=1)

    # Update layout for interactivity and aesthetics
    fig.update_layout(height=800, width=1200, title_text="Vehicle Prices Dashboard", showlegend=False)

    return fig

# Function to visualize model performances
def plot_model_accuracies():
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

    accuracies = [
        [0.38643429, 0.35310009, 0.36801071],  # LinearRegression
        [0.38620243, 0.35350286, 0.36843282],  # Ridge
        [0.38620616, 0.35349711, 0.36843277],  # Lasso
        [0.33686675, 0.31415677, 0.32787848],  # ElasticNet
        [0.62213917, 0.40638212, 0.47242902],  # DecisionTreeRegressor
        [0.74799343, 0.70412406, 0.70161075],  # RandomForestRegressor
        [0.73002938, 0.70887856, 0.70533151],  # GradientBoostingRegressor
        [-0.03261018, -0.05532926, -0.05188942],  # SVR
        [0.64170728, 0.63380643, 0.64356449],  # KNeighborsRegressor
        [-0.38015855, -0.41194531, -0.41229902],  # MLPRegressor
        [0.0021934, -0.43429876, -0.28546934],  # AdaBoostRegressor
        [0.72923447, 0.70932019, 0.67318744],  # BaggingRegressor
        [0.74919345, 0.70561132, 0.68979889]   # ExtraTreesRegressor
    ]

    # Calculate mean accuracies
    mean_accuracies = [sum(acc) / len(acc) for acc in accuracies]

    # Create a bar chart for model accuracies
    fig = px.bar(x=models, y=mean_accuracies, 
                 title='Model Accuracies', 
                 labels={'x': 'Model', 'y': 'Mean Accuracy'},
                 color='mean_accuracy',
                 color_continuous_scale=px.colors.sequential.Blues)

    fig.update_layout(yaxis_range=[-0.5, 1], height=400, width=800)

    return fig

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
            st.markdown(f"### Predicted Price: **${prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Load and display the dataset (replace with actual dataset link)
    df = pd.read_csv('your_dataset.csv')  # Replace with your actual dataset path
    st.write("### Vehicle Price Dataset Preview")
    st.dataframe(df.head())

    # Create and display the dashboard
    dashboard_fig = create_dashboard(df)
    st.plotly_chart(dashboard_fig)

    # Display model accuracy plot
    accuracy_fig = plot_model_accuracies()
    st.plotly_chart(accuracy_fig)

# Run the app
if __name__ == "__main__":
    main()

