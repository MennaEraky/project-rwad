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

def main():
    # Other parts of the main function remain unchanged...

    # Load dataset only once and store in session state
    if 'dataset' not in st.session_state:
        dataset_file_id = '1BMO9pcLUsx970KDTw1kHNkXg2ghGJVBs'  # Replace with your Google Drive file ID for the dataset
        st.session_state.dataset = load_dataset_from_drive(dataset_file_id)

    # Create the dashboard plots
    st.subheader("Dashboard Visualizations")
    if st.session_state.dataset is not None:
        dashboard_fig = create_dashboard(pd.read_csv(st.session_state.dataset))  # Load your dataset for dashboard visualizations
        st.plotly_chart(dashboard_fig)

    # Other parts of the main function remain unchanged...
