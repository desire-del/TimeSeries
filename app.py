import os
import streamlit as st 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils.dataprocess import load_data, df_col, numberOfDiff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.graphics import plotDecompse, plotTs
from models.arima import SARIMAXGridSearch
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
col1, col2 = sider.columns(2)
with col1:
    date_col =sider.selectbox("Date", options=cols)
with col2:
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

# Graphics
st.subheader("Visualize your Time Series")

st.write(df.isna().sum())

obs, decomp, pacf_acf = st.tabs(['Observed', "Decomposition", "PACF AND ACF"])
d , df_diff= numberOfDiff(df.data)
with obs:
    fig1 = plotTs(df)
    st.plotly_chart(fig1)
with decomp:
    fig2 = plotDecompse(df)
    st.plotly_chart(fig2)
    
with pacf_acf:
    fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)
    plot_acf(df_diff,  ax=ax1)
    plot_pacf(df_diff,  ax=ax2)
    st.pyplot(fig)

sider.divider()
# Models selections

options = sider.multiselect(
    'Models',options=["ARIMA"])

for option in options:
    if option == "ARIMA":
        st.divider()
        st.subheader("ARIMA")
        p = st.slider("P", min_value=0, max_value=20, value=1)
        q = st.slider("Q", min_value=0, max_value=20, value=1)
        st.write(d)
        result = None
        if st.checkbox("Turning"):
            pmin = st.slider("P min", min_value=0, max_value=10, value=0)
            pmax = st.slider("P max", min_value=pmin, max_value=pmin+15, value=pmin+10)
            qmin = st.slider("Q min", min_value=0, max_value=10, value=0)
            qmax = st.slider("Q max", min_value=qmin, max_value=qmin+15, value=qmin+10)
            ps = range(pmin, pmax+1)
            ds = range(d, d+1)
            qs = range(qmin, qmax+1)
            if st.button("Find Best Model"):    
                result, best_score, best_param = SARIMAXGridSearch.search(df, ps, ds, qs)
                st.write(result, best_score, best_param)
        else:
            if st.button("Train"):
                model = sm.tsa.ARIMA(df, order=(p,d, q))
                result = model.fit()
        if result:
            with st.expander("Model Diagnostics"):
                st.write(result.plot_diagnostics())
            with st.expander("Model Summary"):
                st.write(result.summary())