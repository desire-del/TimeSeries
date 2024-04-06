import os
import streamlit as st 
import matplotlib.pyplot as plt
from utils.dataprocess import load_data, df_col, numberOfDiff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.graphics import plotDecompse, plotTs
from utils.constants import DEFAULT_DATASETS_DIR

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


# Graphics
st.subheader("Visualize your Time Series")

obs, decomp, pacf_acf = st.tabs(['Observed', "Decomposition", "PACF AND ACF"])
d , df_diff= numberOfDiff(df.data)
with obs:
    fig1 = plotTs(df)
    st.plotly_chart(fig1)
with decomp:
    try:
        fig2 = plotDecompse(df)
        st.plotly_chart(fig2)
    except:
        pass
with pacf_acf:
    fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)
    plot_acf(df_diff,  ax=ax1)
    plot_pacf(df_diff,  ax=ax2)
    st.pyplot(fig)

st.divider()