import streamlit as st
import dill as pickle
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor, StackingRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l1, l2, l1_l2
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# First Streamlit command in the script
st.set_page_config(page_title='Health Disparity Index Across the US', layout='centered')

# Initialize 'predict_clicked' in session state if it doesn't exist
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False

# Set a fixed seed for reproducibility of results
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Function to build a neural network model
@st.cache_resource
def build_model(optimizer='adam', units=100, activation='relu', input_dim=4429, random_state=seed_value):
    if random_state is not None:
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation=activation),
        Dense(units, activation=activation),
        Dense(units, activation=activation),
        Dense(units, activation=activation),
        Dense(units, activation=activation),
        Dense(units, activation=activation),
        Dense(1000, activation=activation),
        Dropout(0.5),
        Dense(units, activation=activation),
        Dense(units, activation=activation, kernel_regularizer=l1_l2(l1=.01, l2=.01)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


# Function to load the model from a .pkl file
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Path to the pickled model
model_path = '/Users/elinarankova/Downloads/Flatiron/Capstone/model/hdi_model.pkl' 
model = load_model(model_path)


# Load and preprocess data
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def load_geo_data():
    # Dummy function to load GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='Geolocation')
    return gdf
    
df = load_data('/Users/elinarankova/Downloads/Flatiron/Capstone/data/final_df.csv')
X = load_data('/Users/elinarankova/Downloads/Flatiron/Capstone/data/X_raw.csv')
df['StateAbbr'] = df['StateAbbr'].str.upper()
df['Geolocationr'] = df['Geolocation'].str.upper()
df['LocationName'] = df['LocationName'].str.title()
df['Geolocation'] = df['Geolocation'].apply(wkt.loads)
gdf = load_geo_data()

# Streamlit page setup
st.title('Health Disparity Index Across the US')
st.image('Images/Banner.jpeg')
st.write('Timespan: 2017-2021')
st.write('### :blue[This program is using data obtained from the PLACES and SDOH data provided by the CDC]')
st.markdown('''
    <div style="text-align: center;">
        To use this platform, simply enter the state abbreviation and county name in the input boxes below.
        The data analyzed will provide insights into health disparities across different regions of the United States.
    </div>
    ''', unsafe_allow_html=True)
st.sidebar.header('Input Features')
state = st.sidebar.selectbox('Select State', df['StateAbbr'].unique())
counties = df[df['StateAbbr'] == state]['LocationName'].unique()
county = st.sidebar.selectbox('Select County', counties)

# Weighted_Idx by Geolocation plot function
@st.cache_data
def geo_plot(feature_str):
    # HTML for custom title styling
    st.markdown("<h5 style='text-align: center; color: black;'>Health Disparity Index by Geolocation</h5>", unsafe_allow_html=True)
    
    # Creating the plot
    fig, ax = plt.subplots(figsize=(20, 12))
    gdf.plot(column=feature_str, ax=ax, legend=True,
             legend_kwds={'label': "Index by Location",
                          'orientation': "horizontal"})
    plt.title('Health Disparity Index by Geolocation')
    plt.xlim(-130, -65)
    plt.ylim(25, 50)
    # Displaying the plot
    st.pyplot(fig)

# Display the plot
geo_plot('Weighted_Idx')

# Setup sidebar and main area
if st.sidebar.button('Predict'):
    st.session_state['predict_clicked'] = True

if st.session_state['predict_clicked']:
    input_data = X[(X['StateAbbr'] == state) & (X['LocationName'] == county)]
    if not input_data.empty:
        prediction_data = input_data.copy()
        try:
            prediction = model.predict(prediction_data)
            st.success(f"Predicted Health Disparity Index: {prediction[0]}")
            st.write("### Detailed Prediction Data")
            st.dataframe(prediction_data)

            #Plot placeholder
        except Exception as e:
            st.error("Failed to make prediction. Ensure your input data is correctly formatted.")
            st.error(f"Error: {e}")
    else:
        st.error("No data available for the selected location. Please select a different location.")
    
    st.session_state['predict_clicked'] = False  # Resetting the predict_clicked state

if st.sidebar.button('Reset'):
    st.session_state['predict_clicked'] = False
    st.experimental_rerun()