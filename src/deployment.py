import os
import streamlit as st
import pickle
import pandas as pd
import json
import geopandas as gpd
from shapely import wkt
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor, StackingRegressor

# First Streamlit command in the script
st.set_page_config(page_title='Health Disparity Across US Counties', layout='centered')

# Initialize 'predict_clicked' in session state if it doesn't exist
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False

# Set a fixed seed for reproducibility of results
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)


# Function to load the model from a .pkl file
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Base directory for models could be set as an environment variable
base_dir = os.getenv('MODEL_BASE_DIR', '/Users/elinarankova/Downloads/Data_Science/Capstone')

# Path to the pickled model
# deploy_directory = os.path.dirname(__file__)
model_path = os.path.join(base_dir,'model','hdi_model.pkl')
# st.write("Trying to load model from:", model_path)
model = load_model(model_path)


# Load and preprocess data
@st.cache_data
def load_data(data_path,dtype_path=None):
    if dtype_path is not None:
            with open(dtype_path, 'r') as f:
                data_types = json.load(f)
    else:
        data_types = None
    df = pd.read_csv(data_path,dtype=data_types)
    return df

@st.cache_data
def load_geo_data(pandas_df):
    # Dummy function to load GeoDataFrame
    geo_df = gpd.GeoDataFrame(pandas_df, geometry='Geometry')
    return geo_df

# Data paths
df_path = os.path.join(base_dir,'data','final_df.csv')
X_path = os.path.join(base_dir,'data','X_raw.csv')
data_type_path = os.path.join(base_dir,'data','data_types.json')
measure_path = os.path.join(base_dir,'data','measure_reference.csv')

df = load_data(df_path)
X = load_data(data_path=X_path, dtype_path=data_type_path)
measure_reference = load_data(measure_path)
df['Geolocation'] = df['Geolocation'].str.upper()
df['Geometry'] = df['Geolocation'].apply(wkt.loads)
gdf = load_geo_data(df)

# Image path
banner_path = os.path.join(base_dir,'images','Banner.jpeg')

# Streamlit page setup
st.title('Health Disparity Across US Counties')
st.image(banner_path)
st.write('#### :blue[This program is using data obtained from the PLACES and SDOH data provided by the CDC]')
st.caption('Timespan: 2017-2021')
st.markdown('''
    <div style="text-align: center;">
        To use this platform, simply select the state abbreviation and counties in the input boxes on the left sidebar.
        The prediction provides insights into health disparities across the selected county of the US.<br><br></div>''', unsafe_allow_html=True)
st.sidebar.header('Input Features')

# Sort states and set them for the selectbox
states = X['StateAbbr'].unique().sort_values().tolist()
state = st.sidebar.selectbox('Select State', states)

# Filter the DataFrame based on the selected state and sort the counties
counties = sorted(X[X['StateAbbr'] == state]['LocationName'].unique())
county = st.sidebar.multiselect('Select County', counties)


# Weighted_Idx by Geolocation plot function
@st.cache_data
def geo_plot(_geo_df,feature_str):
    # HTML for custom title styling
    st.markdown("<h5 style='text-align: center; color: black;'>Health Disparity Index by Geolocation</h5>", unsafe_allow_html=True)
    
    # Creating the plot
    fig, ax = plt.subplots(figsize=(20, 12))
    _geo_df.plot(column=feature_str, ax=ax, legend=True,
             legend_kwds={'label': "Index by Location",
                          'orientation': "horizontal"})
    plt.title('Health Disparity Index by Geolocation')
    plt.xlim(-130, -65)
    plt.ylim(25, 50)
    # Displaying the plot
    st.pyplot(fig)

# Display the plot
geo_plot(gdf,'Weighted_Idx')
st.markdown('''<div style="text-align: center;"><b><i>The larger the index the higher the health disparity in the county.</i></b></div>''', unsafe_allow_html=True)

# Setup sidebar and main area
if st.sidebar.button('Predict'):
    st.session_state['predict_clicked'] = True

if st.session_state['predict_clicked']:
    st.markdown("<ins><h4 style='text-align: center; color: green;'>Prediction Results</h4></ins>", unsafe_allow_html=True)
    input_data = X[(X['StateAbbr'] == state) & (X['LocationName'].isin(county))]
    input_data = input_data.reset_index(drop=True)
    if not input_data.empty:
        prediction_data = input_data.copy()
        try:
            prediction = model.predict(prediction_data).mean()
            st.success(f"Predicted Health Disparity Index: {round(prediction,3)}")
            st.write("##### Detailed Prediction Data")
            with st.expander('Expand for Feature Details'):
                st.markdown('''
                **Category**: General topic within which the measure belongs\n
                **Short Question Text**: Short text summarizing measure\n
                **Total Population**: County population\n
                **Weighted_Idx**: Health Disparity Index
                ''')
            st.caption('This table is interactive! You can search, sort, and download for more in depth analysis.')
            # Defining prediction dataframe
            prediction_df = df[(df['StateAbbr'] == state) & (df['LocationName'].isin(county))]
            prediction_df_filtered = prediction_df.drop(columns=['StateAbbr','LocationID','Scaled_Value', 'Geolocation',
                                                        'GeoPop','PopWeight','Data_Value','Data_Value_Type','Geometry'])

            # Visualizing prediction dataframe
            grouped_df = prediction_df_filtered.groupby(by=['LocationName','Category','Short_Question_Text']).mean().reset_index()
            st.write(grouped_df)
            with st.expander('Expand for Measure Details'):
                st.dataframe(measure_reference)
            
            # Defining and visualizing prediction geolocation
            prediction_gdf = load_geo_data(prediction_df)
            prediction_df.loc[:,'latitude'] = prediction_gdf['Geometry'].y
            prediction_df.loc[:,'longitude'] = prediction_gdf['Geometry'].x
            st.map(prediction_df[['latitude','longitude']],zoom=5)
        except Exception as e:
            st.error("Failed to make prediction. Ensure your input data is correctly formatted.")
            st.error(f"Error: {e}")
    else:
        st.error("No data available for the selected location. Please select a different location.")
    
    st.session_state['predict_clicked'] = False  # Resetting the predict_clicked state

if st.sidebar.button('Reset'):
    st.session_state['predict_clicked'] = False
    st.experimental_rerun()