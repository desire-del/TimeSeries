import xgboost as xgb
from utils.dataprocess import create_features
import streamlit as st 

@st.cache_data
def xgboost(X, y, max_depth, learning_rate, n_estimators):
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X, y)
    return model