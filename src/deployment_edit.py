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
def load_model(model_path):
    st.write("Trying to load the model...")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded inside function.")
    return model

# Path to the pickled model
model_path = '/Users/elinarankova/Downloads/Flatiron/Capstone/model/hdi_model.pkl'  

try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model:")
    st.exception(e)

# Load and preprocess data
df = pd.read_csv('/Users/elinarankova/Downloads/Flatiron/Capstone/data/final_df.csv')
df_proccessed=pd.read_csv('/Users/elinarankova/Downloads/Flatiron/Capstone/data/X_processed.csv')
df['StateAbbr'] = df['StateAbbr'].str.upper()
df['LocationName'] = df['LocationName'].str.title()
df['Geolocation'] = df['Geolocation'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='Geolocation')

# Streamlit page setup
st.title('Health Disparity Index Across the US')
st.image('Images/Banner.jpeg')
st.write('Timespan includes years 2017-2021')
st.write('### :blue[This program is using data obtained from the PLACES and SDOH data provided by the CDC]')
st.markdown('''
    <div style="text-align: center;">
        To use this platform, simply enter the state abbreviation, ZIP code, or county name in the input boxes below.
        The data analyzed will provide insights into health disparities across different regions of the United States.
    </div>
    ''', unsafe_allow_html=True)
st.sidebar.header('User Input Features')

state = st.sidebar.selectbox('Select State', df['StateAbbr'].unique())
counties = df[df['StateAbbr'] == state]['LocationName'].unique()
county = st.sidebar.selectbox('Select County', counties)
zipcode = st.sidebar.text_input('ZIP Code', max_chars=5)

# # Weighted_Idx by Geolocation
# st.markdown("<h5 style='text-align: center; color: black;'>Health Disparity Index by Geolocation</h5>", unsafe_allow_html=True)
# fig, ax = plt.subplots(figsize=(20, 12))
# gdf.plot(column='Weighted_Idx', ax=ax, legend=True,
#                     legend_kwds={'label': "Index by Location",
#                                  'orientation': "horizontal"})
# plt.title('Health Disparity Index by Geolocation')
# plt.xlim(-130, -65)
# plt.ylim(25, 50)
# st.pyplot(fig)

if st.sidebar.button('Predict'):
    st.session_state['predict_clicked'] = True

if st.session_state['predict_clicked']:
    input_data = df[(df['StateAbbr'] == state) & ((df['LocationName'] == county) | (df['ZIPCode'] == zipcode))]
    prediction_data = df_processed.loc[input_data.index]
    prediction = model.predict(prediction_data)
    st.success(f"Predicted Health Disparity Index: {prediction[0]}")
    st.session_state['predict_clicked'] = False

if st.sidebar.button('Reset'):
    st.session_state['predict_clicked'] = False
    st.experimental_rerun()
