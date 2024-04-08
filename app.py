import os
import streamlit as st 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils.dataprocess import load_data, df_col, numberOfDiff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.graphics import plotDecompse, plotTs
from models.arima import SARIMAXGridSearch, white_noise_test, valid_model
from utils.constants import DEFAULT_DATASETS_DIR, FREQ_DICT

# App title
st.title("Time Series Forecasting Web App")

# App slider
sider = st.sidebar

# Load data Section
sider.write("Load Data")
default_data  = sider.selectbox("Defaults", options=os.listdir(DEFAULT_DATASETS_DIR))
upload_file = sider.file_uploader("Uploader un fichier CSV", type=["csv"])
default_url = os.path.join(DEFAULT_DATASETS_DIR, default_data)
data_url =default_url
if upload_file is not None:
    data_url = upload_file
    
cols = df_col(data_url)

date_col =sider.selectbox("Date", options=cols)
data_col = sider.selectbox("Data", options=[c for c in cols if c!=date_col])
try:
    df = load_data(data_url, date_col,data_col)
except:
    cols = df_col(default_url)
    df = load_data(default_url, date_col,data_col)
    st.text("Failed")

# fix the data freq
freq = pd.infer_freq(df.index)
if freq is None:
    st.info("Unable to infer the data freq, enter it manually")
    n = st.number_input(label="Periode of ", min_value=1, step=1, format='%u')
    freq_type = st.selectbox("Type", options=list(FREQ_DICT.keys()))
    df = df.asfreq(freq=str(n)+FREQ_DICT[freq_type], method='bfill')
else:
    df = df.asfreq(freq=freq, method='bfill')

# Save a copy of the dataframe
df_copy = df.copy()
# Data Transformation
sider.divider()
sider.write("Data tranformation")
apply_log = sider.toggle("Apply log")
if apply_log:
    df["data"]  = np.log(df["data"])
    df = df.dropna()
# Graphics
st.subheader("Visualize your Time Series")

obs, decomp, season,pacf_acf = st.tabs(['Observed', "Decomposition","SEASONALITY","PACF AND ACF"])
d , df_diff= numberOfDiff(df.data)
with obs:
    fig1 = plotTs(df)
    st.plotly_chart(fig1)
with decomp:
    fig2 = plotDecompse(df)
    st.plotly_chart(fig2)
with season:
    fig1, ax = plt.subplots()
    lags = st.number_input(label="Lags ", min_value=25, max_value=len(df)-1, step=5, format='%u')
    plot_acf(df, lags=lags, ax=ax)
    st.pyplot(fig1)
with pacf_acf:
    fig2, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)
    plot_acf(df,  ax=ax1)
    plot_pacf(df,  ax=ax2)
    st.pyplot(fig2)

sider.divider()
# Models selections

options = sider.multiselect(
    'Models',options=["ARIMA", "SARIMA"])

for option in options:
    if option == "ARIMA":
        st.divider()
        sider.divider()
        sider.subheader("ARIMA")
        st.subheader("ARIMA")
        result = None
        p_range = sider.slider("P Range",min_value=0, max_value=30, value=[0, 0])
        q_range = sider.slider("Q Range",min_value=0, max_value=30, value=[0, 0])
        ps = range(p_range[0], p_range[1]+1)
        ds = range(d, d+1)
        qs = range(q_range[0], q_range[1]+1)
        if sider.button("Train"):    
            result, best_score, best_param = SARIMAXGridSearch.search(df, ps, ds, qs)
            st.write(result, best_score, best_param)
        if result:
            with st.expander("Model Diagnostics"):
                st.write(result.plot_diagnostics())
            with st.expander("Model Validation"):
                lb = float(result.summary().tables[2].data[1][1])
                jb = float(result.summary().tables[2].data[1][3])
                st.markdown(f'<h3>Ljung-Box Test : {lb}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h3>Jarque-Box Test : {jb}</h3>', unsafe_allow_html=True)
                color, label = ("green", "Validate") if valid_model(lb, jb) else ("red", "Reject")
                st.markdown(f'<h3 style="color:{color};">Decision : {label}</h3>', unsafe_allow_html=True)
                
    if option == "SARIMA":
        continue