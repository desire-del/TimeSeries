import pandas as pd 
import streamlit as st 
import statsmodels.api as sm
from itertools import product
from stqdm import stqdm
"""
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
"""
class SARIMAXGridSearch:
    def __init__(self, endog):
        pass
        
    @staticmethod
    @st.cache_data
    def search(endog, p_range, d_range, q_range):
        best_score = float('inf')
        best_param = None
        best_result = None
        order_combination = list(product(p_range, d_range, q_range))
        
        for order in stqdm(order_combination):
            try: 
                model = sm.tsa.SARIMAX(endog, order=order)
                result = model.fit(disp=False)
                if result.aic < best_score:
                    best_score = result.aic 
                    best_param = order
                    best_result = result
            except:
                continue 
        return best_result, best_score, best_param
            
    