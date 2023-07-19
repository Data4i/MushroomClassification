import streamlit as st

import pickle
import pandas as pd

@st.cache_resource
def get_resources(std_scaler_filename, model_filename):
    model = pickle.load(open(model_filename, 'rb'))
    scaler = pickle.load(open(std_scaler_filename, 'rb'))
    return scaler, model

@st.cache_data
def get_data(data_path):
    return pd.read_csv(data_path)
