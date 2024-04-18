import streamlit as st
import dill as pickle
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import random
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
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


# Add function for pickle loading capabilities
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

# Load model with debugging
st.write("App started - This should always appear when you reload the app")
model_path = '/Users/elinarankova/Downloads/Flatiron/Capstone/model/hdi_model.pkl'  

# Function to load the model
def load_model(model_path):
    st.write("Trying to load the model...")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded inside function.")
    return model

try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")  # Should appear if model loads without issues
except Exception as e:
    st.error("Failed to load model:")
    st.exception(e)  # This will print the exception with a traceback
    
# Load data
df=pd.read_csv('/Users/elinarankova/Downloads/Flatiron/Capstone/data/final_df.csv')
df_proccessed=pd.read_csv('/Users/elinarankova/Downloads/Flatiron/Capstone/data/X_processed.csv')

# Data preproceessing
df['StateAbbr'] = df['StateAbbr'].str.upper()
df['Geolocation'] = df['Geolocation'].str.upper()
df['LocationName'] = df['LocationName'].str.title()
df.loc[:,'Geolocation'] = df['Geolocation'].apply(wkt.loads)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=df['Geolocation'])

# Streamlit page configuration
st.set_page_config(page_title='Health Disparity Index Across the US', layout='centered')


# Initialize session state for the predict button
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False


# Sidebar for user inputs
st.sidebar.header('Input Features')
state = st.sidebar.selectbox('State', df['StateAbbr'].unique())

# Ensuring that counties corresponding to the selected state are displayed
counties = df.loc[df['StateAbbr'] == state, 'LocationName'].unique()
county = st.sidebar.selectbox('County', counties if counties.size > 0 else ['No counties available'])
zipcode = st.sidebar.text_input('ZIP Code', max_chars=5)


# Displaying selected inputs
st.markdown("<h5 style='text-align: center; color: black;'>Selected Location</h5>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**State:** {state}")
with col2:
    st.write(f"**County:** {county}")
with col3:
    st.write(f"**5 Digit ZIP Code:** {zipcode}")


# Predict button in the sidebar
if st.sidebar.button('Predict'):
    st.session_state['predict_clicked'] = True

# Main area
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

# Weighted_Idx by Geolocation
if 'gdf' in locals():
    st.markdown("<h5 style='text-align: center; color: black;'>Health Disparity Index by Geolocation</h5>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 12))
    gdf.plot(column='Weighted_Idx', ax=ax, legend=True,
                        legend_kwds={'label': "Index by Location",
                                     'orientation': "horizontal"})
    plt.title('Health Disparity Index by Geolocation')
    plt.xlim(-130, -65)
    plt.ylim(25, 50)
    st.pyplot(fig)

# Handling prediction logic
if st.session_state['predict_clicked']:
    input_data = df[(df['StateAbbr'] == state) & ((df['LocationName'] == county) | (df['LocationID'] == zipcode))]
    input_geo_data = gdf[(gdf['StateAbbr'] == state) & ((gdf['LocationName'] == county) | (gdf['LocationID'] == zipcode))]
    prediction_data = df_processed.loc[input_data.index]
    
    if not input_data.empty:
        prediction = model.predict(prediction_data)
        st.success(f"The predicted health disparity index is: {prediction[0]}")
        
        # Plotting the geographic data
        st.header('Health Disparity Index by Geolocation')
        fig, ax = plt.subplots(figsize=(20, 10))
        input_geo_data.plot(column='Weighted_Idx', ax=ax, legend=True,
                            legend_kwds={'label': "Index by Location", 'orientation': "horizontal"})
        plt.title('Health Disparity Index by Geolocation')
        st.pyplot(fig)
        
        # Additional Data Display
        st.header('Population by Geolocation')
        fig, ax = plt.subplots(figsize=(20, 10))
        input_geo_data.plot(column='TotalPopulation', ax=ax, legend=True)
        plt.title('Population by Geolocation')
        st.pyplot(fig)
        
        # Show detailed data in a table
        st.dataframe(input_data[['Category', 'Short_Question_Text', 'Data_Value', 'TotalPopulation']])
    else:
        st.error("No data available for the selected location.")
    
    # Reset the predict button state to allow for re-runs
    st.session_state['predict_clicked'] = False