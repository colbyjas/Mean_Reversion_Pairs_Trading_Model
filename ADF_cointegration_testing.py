# -*- coding: utf-8 -*-
"""
Created on Fri May 16 00:14:28 2025

@author: Colby Jaskowiak
"""
#%%

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller, coint

#%%

tickers = ['XOM', 'CVX', 'COP', 'OXY', 'PSX', 'SLB', 'HAL', 'VLO', 'MPC', 'HES']

price_data = yf.download(tickers, start="2018-01-01", end="2024-12-31")['Close'].dropna()

results = []

for stock_a, stock_b in combinations(tickers, 2):
    pair_name = f"{stock_a}-{stock_b}"
    pair_data = price_data[[stock_a, stock_b]].dropna()

    log_a = np.log(pair_data[stock_a])
    log_b = np.log(pair_data[stock_b])
    spread = log_a - log_b

    adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread)

    coint_stat, coint_pvalue, _ = coint(log_a, log_b)

    results.append({
        'Pair': pair_name,
        'ADF_pvalue': adf_pvalue,
        'Cointegration_pvalue': coint_pvalue
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['ADF_pvalue', 'Cointegration_pvalue']).reset_index(drop=True)
print(results_df.head(10))

#%%

tickers = [
    # Consumer Staples
    'KO', 'PEP', 'PG', 'CL', 'WMT', 'COST',
    # Industrials
    'UNP', 'CSX', 'CAT', 'DE', 'MMM', 'GE'
]


price_data = yf.download(tickers, start="2020-05-13", end="2025-05-13")['Close'].dropna()

results = []

for stock_a, stock_b in combinations(tickers, 2):
    pair_data = price_data[[stock_a, stock_b]].dropna()
    log_a = np.log(pair_data[stock_a])
    log_b = np.log(pair_data[stock_b])
    spread = log_a - log_b

    
    adf_p = adfuller(spread)[1]
    coint_p = coint(log_a, log_b)[1]

    results.append({
        'Pair': f"{stock_a}-{stock_b}",
        'ADF_pvalue': adf_p,
        'Cointegration_pvalue': coint_p
    })

# Show top results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['ADF_pvalue', 'Cointegration_pvalue']).reset_index(drop=True)
print(results_df.head(15))

#%%
