import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Function to load a trained model
def load_model():
    with open('vehicle_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to visualize feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create a bar chart for feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Vehicle Price Prediction - Feature Importance Analysis")

    # Load the model
    model = load_model()

    # Load your dataset
    df = pd.read_csv('vehicle_data.csv')  # Update with your dataset path
    X = df.drop(columns=['price'])  # Assuming 'price' is the target variable
    y = df['price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display feature importance
    st.subheader("Feature Importance")
    plot_feature_importance(model, X.columns)

    # Allow user to select the number of features
    num_features = st.slider("Select number of features to include in model:", 1, X.shape[1], 5)

    # Get the top features based on importance
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    top_features = feature_importances.nlargest(num_features, 'Importance')

    st.subheader(f"Top {num_features} Features")
    st.write(top_features)

    # Visualize selected features
    st.subheader("Visualization of Selected Features")
    for feature in top_features['Feature']:
        plt.figure()
        plt.scatter(df[feature], df['price'], alpha=0.5)
        plt.title(f'{feature} vs Price')
        plt.xlabel(feature)
        plt.ylabel('Price')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
