import pandas as pd 
import streamlit as st 
import statsmodels.api as sm

@st.cache_data
def gridSearch(endog, order_ls:list, d= 0):
    res = []
    for order in order_ls: 
        try:
            model = sm.tsa.SARIMAX(endog, order=(order[0], d, order[1]))
            model_fit = model.fit(disp=False)
            aic = model_fit.aic
            res.append([(order[0], d,order[1]), aic])
        except:
            continue
    res_df = pd.DataFrame(res, columns=["pdq", "AIC"])
    res_df = res_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)
    return res_df