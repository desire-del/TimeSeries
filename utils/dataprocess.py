import pandas as pd
import numpy as np
from .constants import FREQ_DICT
import streamlit as st
from statsmodels.tsa.stattools import adfuller, acf

@st.cache_data
def df_col(file_url):
    df = pd.read_csv(file_url)
    return df.columns
@st.cache_data
def load_data(file_url, date_col, data_col):
    df = pd.read_csv(file_url)
    # Select date and data columns
    df = df[[date_col, data_col]]
    df.columns = ["date", "data"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df 

@st.cache_data
def isStatinary(y):
    res = adfuller(y)
    return True if res[1] <= 0.05 else False

@st.cache_data
def numberOfDiff(y):
    diff = y
    d = 0
    while ((not isStatinary(diff)) and d < 5):
        diff = np.diff(diff)
        d = d+1
    return d, diff