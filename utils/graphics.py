import streamlit as st 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.express as px
import pandas as pd

@st.cache_data
def plotTs(df):
    fig = px.line(df, x=df.index, y=df.columns)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 Day", step="day", stepmode="backward"),
                dict(count=1, label="1 Month", step="month", stepmode="backward"),
                dict(count=3, label="3 Month", step="month", stepmode="backward"),
                dict(count=1, label="1 Year", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig

@st.cache_data
def plotDecompse(df):
    decomp = sm.tsa.STL(df)
    decomp =decomp.fit()
    resid = decomp.resid
    trend = decomp.trend
    seasonal = decomp.seasonal
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Line(x = trend.index, y=trend, name = "Trend"),
        row=1,
        col=1,
        
    )
    fig.add_trace(
        go.Line(x = seasonal.index, y = seasonal, name = "Seasonal"),
        row=2,
        col=1,
        
    )
    fig.add_trace(
        go.Line(x = resid.index, y= resid, name = "Residual"),
        row = 3,
        col = 1,
        
    )
    return fig

@st.cache_data
def plotForcast(df, pred, confint):
    fig = go.Figure()
    fig.add_trace(
        go.Line(
            x = df.index,
            y = df.data,
            name= "Observed"
        )
    )
    fig.add_trace(
        go.Line(
            x = pred.index,
            y = pred,
            name="Prediction",
            line=dict(color='rgba(255,0,0,1)'),
        )
    )
    fig.add_trace(
        go.Scatter(
            x =confint.index,
            y = confint["lower data"],
            mode="lines",
            line=dict(color='rgba(0,100,80,0.2)'),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x =confint.index,
            y = confint["upper data"],
            mode="lines",
            line=dict(color='rgba(0,100,80,0.2)'),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x = confint.index.union(confint.index[::-1]),
            y = pd.concat([confint["lower data"],confint["upper data"][::-1]]),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        )
    )
    
    return fig