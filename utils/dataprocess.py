import pandas as pd
import numpy as np
from .constants import FREQ_DICT, P_VALUE_THRESHOLD
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
    return True if res[1] <= P_VALUE_THRESHOLD else False

@st.cache_data
def numberOfDiff(y):
    diff = y
    d = 0
    while ((not isStatinary(diff)) and d < 5):
        diff = np.diff(diff)
        d = d+1
    return d, diff

@st.cache_data
def create_features(df, lags = 1):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    for i in range(1, lags):
        df[f'lag_{i}'] = df["data"].shift(i)
    df["target"] = df["data"].shift(lags)
    df.dropna(inplace=True)
    y = df[["target"]]
    X = df.drop(columns=["target", "date"])
    return X, y