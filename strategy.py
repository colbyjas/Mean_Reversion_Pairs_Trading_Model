# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:11:59 2025

@author: Colby Jaskowiak
"""

# Strategy 1 implementation file
# Pairs Trading Strategy

#%%

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# Data Import and Preprocessing Historical Close Prices

tickers = ['XOM', 'VLO', 'PG', 'UNP']

price_data = yf.download(tickers, start='2020-05-13', end=pd.Timestamp.today().strftime('%Y-%m-%d'))['Close']
price_data.dropna(inplace=True)

# Create Pairs
Pair_1 = price_data[['XOM', 'VLO']].copy()
Pair_2 = price_data[['PG', 'UNP']].copy()

#%%
# Log Prices and log Spread

Pair_1_log = np.log(Pair_1)
Pair_2_log = np.log(Pair_2)

Pair_1_log['Spread'] = Pair_1_log['XOM'] - Pair_1_log['VLO']
Pair_2_log['Spread'] = Pair_2_log['PG'] - Pair_2_log['UNP']

#%%
# Plot Prices and Spread

def plot_pair(log_df, label_a, label_b, title):
    fig, axs = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    axs[0].plot(log_df.index, log_df[label_a], label=label_a)
    axs[0].plot(log_df.index, log_df[label_b], label=label_b)
    axs[0].set_title(f'{title} - Log Prices')
    axs[0].legend()

    axs[1].plot(log_df.index, log_df['Spread'], color='purple')
    axs[1].set_title('Log Price Spread')
    axs[1].axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.show()

# Plot both pairs
plot_pair(Pair_1_log, 'XOM', 'VLO', 'XOM vs VLO')
plot_pair(Pair_2_log, 'PG', 'UNP', 'PG vs UNP')

#%%
# Normalized Prices to compare relative performance

normalized = price_data / price_data.iloc[0]
normalized.plot(figsize=(12,6), title='Normalized Prices')

#%%
# Stationary and Cointegration Testing

from statsmodels.tsa.stattools import adfuller, coint

#%%
# Stationary Testing

def test_adf(spread, pair_name):
    result = adfuller(spread)
    print(f"ADF Test for {pair_name} Spread")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value:.4f}")
    print("-" * 40)

test_adf(Pair_1_log['Spread'], "XOM-VLO")
test_adf(Pair_2_log['Spread'], "PG-UNP")

#%%
# Cointegration Testing

def test_cointegration(series_a, series_b, pair_name):
    score, pvalue, _ = coint(series_a, series_b)
    print(f"Cointegration Test for {pair_name}")
    print(f"Test Statistic: {score:.4f}")
    print(f"p-value: {pvalue:.4f}")
    print("-" * 40)

test_cointegration(Pair_1_log['XOM'], Pair_1_log['VLO'], "XOM-VLO")
test_cointegration(Pair_2_log['PG'], Pair_2_log['UNP'], "PG-UNP")

#%%
# Generating Z-Score and Trading Signals

def generate_zscore_signals(log_df, lookback=60, entry_threshold=1.0, exit_threshold=0.0):
    spread = log_df['Spread']
    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std()
    zscore = (spread - mean) / std

    signal = np.where(zscore > entry_threshold, -1,  # Short spread
              np.where(zscore < -entry_threshold, 1,  # Long spread
              np.where(abs(zscore) < exit_threshold, 0, np.nan)))  # Exit

    signal = pd.Series(signal, index=log_df.index).ffill().fillna(0)

    log_df['ZScore'] = zscore
    log_df['Signal'] = signal
    return log_df

Pair_1_log = generate_zscore_signals(Pair_1_log.copy())  # XOM-VLO
Pair_2_log = generate_zscore_signals(Pair_2_log.copy())  # PG-UNP

#%%
# Cleaning NaN in Columns
Pair_1_log_clean = Pair_1_log.dropna(subset=['ZScore', 'Signal']).copy()
Pair_2_log_clean = Pair_2_log.dropna(subset=['ZScore', 'Signal']).copy()

aligned_dates = Pair_1_log_clean.index.intersection(Pair_2_log_clean.index)

Pair_1_log_aligned = Pair_1_log_clean.loc[aligned_dates].copy()
Pair_2_log_aligned = Pair_2_log_clean.loc[aligned_dates].copy()

#%%
# Backtesting XOM-VLO strategy

# Undo log transformation to get prices back
price_xom = np.exp(Pair_1_log['XOM'])
price_vlo = np.exp(Pair_1_log['VLO'])

# Calculate daily returns
ret_xom = price_xom.pct_change()
ret_vlo = price_vlo.pct_change()

# Strategy return: Signal determines long/short side
strategy_ret_1 = Pair_1_log['Signal'].shift(1) * (ret_xom - ret_vlo)

cumulative_ret_1 = (1 + strategy_ret_1).cumprod()

#%%
# XOM-VLO Cumulative Returns Chart
plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_1, label='Strategy Cumulative Return (XOM-VLO)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('XOM-VLO Pairs Trading Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Backtesting PG-UNP strategy

price_pg = np.exp(Pair_2_log['PG'])
price_unp = np.exp(Pair_2_log['UNP'])

ret_pg = price_pg.pct_change()
ret_unp = price_unp.pct_change()

strategy_ret_2 = Pair_2_log['Signal'].shift(1) * (ret_pg - ret_unp)

cumulative_ret_2 = (1 + strategy_ret_2).cumprod()

#%%
# PG-UNP Cumilative Returns Chart
plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_2, label='Strategy Cumulative Return (PG-UNP)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('PG-UNP Pairs Trading Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# Validation Testing using split data XOM-VLO (Out-of-sample data testing)

# Split in/out sample
split_date = '2022-12-31'
train_data_1 = Pair_1.loc[:split_date]
test_data_1 = Pair_1.loc[split_date:]

#%%

# In-sample log spread
train_log_1 = np.log(train_data_1)
train_log_1['Spread_1'] = train_log_1['XOM'] - train_log_1['VLO']
mean_1 = train_log_1['Spread_1'].rolling(window=60).mean().dropna().iloc[-1]
std_1 = train_log_1['Spread_1'].rolling(window=60).std().dropna().iloc[-1]

# Out-of-sample log spread and z-score
test_log_1 = np.log(test_data_1)
test_log_1['Spread_1'] = test_log_1['XOM'] - test_log_1['VLO']
test_log_1['ZScore_1'] = (test_log_1['Spread_1'] - mean_1) / std_1

#%%
# Signals
entry_threshold = 1.0
exit_threshold = 0.0
signal_1 = np.where(test_log_1['ZScore_1'] > entry_threshold, -1,
            np.where(test_log_1['ZScore_1'] < -entry_threshold, 1,
            np.where(abs(test_log_1['ZScore_1']) < exit_threshold, 0, np.nan)))
test_log_1['Signal_1'] = pd.Series(signal_1, index=test_log_1.index).ffill().fillna(0)

#%%
# Backtest
price_xom_1 = np.exp(test_log_1['XOM'])
price_vlo_1 = np.exp(test_log_1['VLO'])
ret_xom_1 = price_xom_1.pct_change()
ret_vlo_1 = price_vlo_1.pct_change()

strategy_ret_oos_1 = test_log_1['Signal_1'].shift(1) * (ret_xom_1 - ret_vlo_1)
cumulative_ret_oos_1 = (1 + strategy_ret_oos_1).cumprod()

#%%
# Out of Sample Test XOM-VLO
plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_oos_1, label='Out-of-Sample Return (XOM-VLO)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Out-of-Sample Strategy Performance: XOM-VLO')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Validation Testing using split data PG-UNP (Out-of-sample data testing)

split_date = '2022-12-31'
train_data_2 = Pair_2.loc[:split_date]
test_data_2 = Pair_2.loc[split_date:]

#%%

train_log_2 = np.log(train_data_2)
train_log_2['Spread_2'] = train_log_2['PG'] - train_log_2['UNP']
mean_2 = train_log_2['Spread_2'].rolling(window=60).mean().dropna().iloc[-1]
std_2 = train_log_2['Spread_2'].rolling(window=60).std().dropna().iloc[-1]

test_log_2 = np.log(test_data_2)
test_log_2['Spread_2'] = test_log_2['PG'] - test_log_2['UNP']
test_log_2['ZScore_2'] = (test_log_2['Spread_2'] - mean_2) / std_2

#%%
# Signals
entry_threshold = 1.0
exit_threshold = 0.0
signal_2 = np.where(test_log_2['ZScore_2'] > entry_threshold, -1,
            np.where(test_log_2['ZScore_2'] < -entry_threshold, 1,
            np.where(abs(test_log_2['ZScore_2']) < exit_threshold, 0, np.nan)))
test_log_2['Signal_2'] = pd.Series(signal_2, index=test_log_2.index).ffill().fillna(0)

#%%
# Backtest
price_pg_2 = np.exp(test_log_2['PG'])
price_unp_2 = np.exp(test_log_2['UNP'])
ret_pg_2 = price_pg_2.pct_change()
ret_unp_2 = price_unp_2.pct_change()

strategy_ret_oos_2 = test_log_2['Signal_2'].shift(1) * (ret_pg_2 - ret_unp_2)
cumulative_ret_oos_2 = (1 + strategy_ret_oos_2).cumprod()

#%%
# Out of Sample Test PG-UNP
plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_oos_2, label='Out-of-Sample Return (PG-UNP)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Out-of-Sample Strategy Performance: PG-UNP')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Adding Transaction Costs for XOM-VLO

cost_per_trade = 0.002  # 0.2%
signal_change_1 = test_log_1['Signal_1'].diff().abs() > 0
cost_impact_1 = signal_change_1 * cost_per_trade
strategy_ret_oos_cost_adj_1 = strategy_ret_oos_1 - cost_impact_1
cumulative_ret_oos_cost_adj_1 = (1 + strategy_ret_oos_cost_adj_1).cumprod()

#%%
# Plotting Returns with Transaction Costs XOM-VLO

plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_oos_1, label='Without Transaction Costs')
plt.plot(cumulative_ret_oos_cost_adj_1, label='With Transaction Costs', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.title('XOM-VLO: Out-of-Sample Return (With vs. Without Costs)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Addining Transaction Costs PG-UNP

cost_per_trade = 0.002
signal_change_2 = test_log_2['Signal_2'].diff().abs() > 0
cost_impact_2 = signal_change_2 * cost_per_trade
strategy_ret_oos_cost_adj_2 = strategy_ret_oos_2 - cost_impact_2
cumulative_ret_oos_cost_adj_2 = (1 + strategy_ret_oos_cost_adj_2).cumprod()

#%%
# Plotting Returns with Transaction Costs PG-UNP

plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_oos_2, label='Without Transaction Costs')
plt.plot(cumulative_ret_oos_cost_adj_2, label='With Transaction Costs', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.title('PG-UNP: Out-of-Sample Return (With vs. Without Costs)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Volatility Scaling for XOM-VLO

test_log_1['RollingStd_1'] = test_log_1['Spread_1'].rolling(20).std()
inv_vol = 1 / test_log_1['RollingStd_1']
scaled_weight = (inv_vol - inv_vol.min()) / (inv_vol.max() - inv_vol.min())
scaled_weight = 0.5 + scaled_weight * 1.0

test_log_1['ScaledSignal_1'] = test_log_1['Signal_1'] * scaled_weight

#%%
# Calculate Returns based on new volatility scaling XOM-VLO

strategy_ret_unscaled_1 = test_log_1['Signal_1'].shift(1) * (ret_xom_1 - ret_vlo_1)
strategy_ret_scaled_1 = test_log_1['ScaledSignal_1'].shift(1) * (ret_xom_1 - ret_vlo_1)

cumulative_ret_unscaled_1 = (1 + strategy_ret_unscaled_1).cumprod()
cumulative_ret_scaled_1 = (1 + strategy_ret_scaled_1).cumprod()

#%%
# Chart New Returns XOM-VLO

plt.figure(figsize=(12,6))
plt.plot(cumulative_ret_unscaled_1, label='Unscaled Signal')
plt.plot(cumulative_ret_scaled_1, label='Volatility-Scaled Signal', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.title('XOM-VLO: Cumulative Return (Volatility Scaling Comparison)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Volatility Scaling for PG-UNP

test_log_2['RollingStd_2'] = test_log_2['Spread_2'].rolling(20).std()
inv_vol_2 = 1 / test_log_2['RollingStd_2']
scaled_weight_2 = (inv_vol_2 - inv_vol_2.min()) / (inv_vol_2.max() - inv_vol_2.min())
scaled_weight_2 = 0.5 + scaled_weight_2 * 1.0

test_log_2['ScaledSignal_2'] = test_log_2['Signal_2'] * scaled_weight_2

#%%

strategy_ret_unscaled_2 = test_log_2['Signal_2'].shift(1) * (ret_pg_2 - ret_unp_2)
strategy_ret_scaled_2 = test_log_2['ScaledSignal_2'].shift(1) * (ret_pg_2 - ret_unp_2)

cumulative_ret_unscaled_2 = (1 + strategy_ret_unscaled_2).cumprod()
cumulative_ret_scaled_2 = (1 + strategy_ret_scaled_2).cumprod()

#%%

plt.figure(figsize=(12,6))
plt.plot(cumulative_ret_unscaled_2, label='Unscaled Signal')
plt.plot(cumulative_ret_scaled_2, label='Volatility-Scaled Signal', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.title('PG-UNP: Cumulative Return (Volatility Scaling Comparison)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Calculating Sharpe Ratio for both pairs to determine if risk-adjusted performance improved

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_ret = returns - risk_free_rate / periods_per_year
    return (excess_ret.mean() / excess_ret.std()) * np.sqrt(periods_per_year)

# XOM-VLO
sharpe_unscaled_1 = sharpe_ratio(strategy_ret_unscaled_1)
sharpe_scaled_1 = sharpe_ratio(strategy_ret_scaled_1)

# PG-UNP
sharpe_unscaled_2 = sharpe_ratio(strategy_ret_unscaled_2)
sharpe_scaled_2 = sharpe_ratio(strategy_ret_scaled_2)

print(f"XOM-VLO Sharpe (Unscaled): {sharpe_unscaled_1:.2f}")
print(f"XOM-VLO Sharpe (Scaled):   {sharpe_scaled_1:.2f}")
print("-" * 40)
print(f"PG-UNP Sharpe (Unscaled):  {sharpe_unscaled_2:.2f}")
print(f"PG-UNP Sharpe (Scaled):    {sharpe_scaled_2:.2f}")

#%%
# Risk-Based Allocation Logic (Rolling Inverse Volatility)

vol_1 = strategy_ret_unscaled_1.rolling(window=20).std()
vol_2 = strategy_ret_unscaled_2.rolling(window=20).std()

inv_vol_1 = 1 / vol_1
inv_vol_2 = 1 / vol_2

total_inv_vol = inv_vol_1 + inv_vol_2
weight_1 = inv_vol_1 / total_inv_vol
weight_2 = inv_vol_2 / total_inv_vol

combined_return = weight_1 * strategy_ret_unscaled_1 + weight_2 * strategy_ret_unscaled_2
combined_cumulative_return = (1 + combined_return).cumprod()

#%%
# Total Portfolio Returns Plot with Inverse Volatility Weighting

plt.figure(figsize=(12,6))
plt.plot(combined_cumulative_return, label='Risk-Weighted Portfolio Return', color='darkgreen')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Combined Strategy: Cumulative Return with Inverse Volatility Weighting (Unscaled Returns)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Sharpe Ratio for Combined portfolio

sharpe_combined = combined_return.mean() / combined_return.std() * np.sqrt(252)
print(f"Combined Portfolio Sharpe Ratio: {sharpe_combined:.2f}")

#%%
# Adding Macro and Style Factor Filters

# 1. Momentum Filter using RSI

# Compute RSI for spread

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain, index=series.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#%%
# XOM-VLO RSI Calculation
# Apply RSI to Spread
test_log_1['RSI_1'] = compute_rsi(test_log_1['Spread_1'])

# Create binary mask for neutral RSI range
test_log_1['RSI_Mask_1'] = ((test_log_1['RSI_1'] > 30) & (test_log_1['RSI_1'] < 70)).astype(int)

# Filter original signal
test_log_1['FilteredSignal_1'] = test_log_1['Signal_1'] * test_log_1['RSI_Mask_1']

#%%
# PG-UNP RSI Calculation

test_log_2['RSI_2'] = compute_rsi(test_log_2['Spread_2'])

test_log_2['RSI_Mask_2'] = ((test_log_2['RSI_2'] > 30) & (test_log_2['RSI_2'] < 70)).astype(int)

test_log_2['FilteredSignal_2'] = test_log_2['Signal_2'] * test_log_2['RSI_Mask_2']

#%%

# Computing returns using RSI Filter for both pairs

# XOM-VLO
strategy_ret_filtered_1 = test_log_1['FilteredSignal_1'].shift(1) * (ret_xom - ret_vlo)
cumulative_ret_filtered_1 = (1 + strategy_ret_filtered_1).cumprod()

# PG-UNP
price_pg = np.exp(test_log_2['PG'])
price_unp = np.exp(test_log_2['UNP'])
ret_pg = price_pg.pct_change()
ret_unp = price_unp.pct_change()

strategy_ret_filtered_2 = test_log_2['FilteredSignal_2'].shift(1) * (ret_pg - ret_unp)
cumulative_ret_filtered_2 = (1 + strategy_ret_filtered_2).cumprod()

#%%
# Plotting Fitlered Returns

plt.figure(figsize=(12,6))
plt.plot(cumulative_ret_filtered_1, label='XOM-VLO (RSI Filtered)')
plt.plot(cumulative_ret_filtered_2, label='PG-UNP (RSI Filtered)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('RSI-Filtered Strategy: Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Combine RSI-Filtered Returns for Both Pairs (Out-of-Sample)
# Make sure the filtered strategy returns for both pairs exist:
# strategy_ret_filtered_1 = XOM-VLO filtered returns
# strategy_ret_filtered_2 = PG-UNP filtered returns

# Combine equally weighted
combined_filtered_ret = 0.5 * strategy_ret_filtered_1 + 0.5 * strategy_ret_filtered_2
combined_filtered_cumulative = (1 + combined_filtered_ret).cumprod()

#%%
# Plot Combined Cumulative Returns (RSI-Filtered)

plt.figure(figsize=(12,6))
plt.plot(combined_filtered_cumulative, label='RSI-Filtered Portfolio Return')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Combined Portfolio Cumulative Return (RSI-Filtered)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% 
# Apply Risk-Based Allocation to RSI-Filtered Returns

# Ensure weights are aligned with filtered returns (already computed earlier)
# weight_1 and weight_2 come from rolling inverse vol on unscaled returns

combined_filtered_risk_weighted_ret = weight_1 * strategy_ret_filtered_1 + weight_2 * strategy_ret_filtered_2
combined_filtered_risk_weighted_cumret = (1 + combined_filtered_risk_weighted_ret).cumprod()

#%% 
# Plot Risk-Based Allocated RSI-Filtered Portfolio Returns

plt.figure(figsize=(12,6))
plt.plot(combined_filtered_risk_weighted_cumret, label='RSI-Filtered (Risk-Based Allocated)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Combined Portfolio Return (RSI-Filtered + Risk Allocation)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%

#2. Volatility Regime Filter (VIX)

vix_data = yf.download('^VIX', start='2020-05-13', end=pd.Timestamp.today().strftime('%Y-%m-%d'))['Close']
vix_data.name = 'VIX'
vix_data = vix_data.ffill()

#%% 
# Define VIX Threshold and Binary Mask
vix_threshold = 20
vix_mask = (vix_data < vix_threshold).astype(int)
vix_mask.name = "VIX_Filter"

# Combine for inspection
vix_mask_df = pd.concat([vix_data, vix_mask], axis=1)

#%% 
# 3. Apply VIX Filter to Signals

# Align VIX mask and ensure it's a Series
vix_mask_aligned_1 = vix_mask.reindex(test_log_1.index).ffill()
vix_mask_aligned_2 = vix_mask.reindex(test_log_2.index).ffill()

# Ensure Series shape (avoid DataFrame shape)
vix_mask_aligned_1 = vix_mask_aligned_1.squeeze()
vix_mask_aligned_2 = vix_mask_aligned_2.squeeze()

# Multiply filtered RSI signals with VIX mask
test_log_1['Signal_RSI_VIX_1'] = test_log_1['FilteredSignal_1'] * vix_mask_aligned_1
test_log_2['Signal_RSI_VIX_2'] = test_log_2['FilteredSignal_2'] * vix_mask_aligned_2

#%%
# Computing Returns with RSI + VIX Filter
# XOM-VLO
strategy_ret_rsi_vix_1 = test_log_1['Signal_RSI_VIX_1'].shift(1) * (ret_xom - ret_vlo)
cumulative_ret_rsi_vix_1 = (1 + strategy_ret_rsi_vix_1).cumprod()

# PG-UNP
strategy_ret_rsi_vix_2 = test_log_2['Signal_RSI_VIX_2'].shift(1) * (ret_pg - ret_unp)
cumulative_ret_rsi_vix_2 = (1 + strategy_ret_rsi_vix_2).cumprod()

#%%
# Plotting Returns with RSI + VIX Filter
plt.figure(figsize=(12, 6))
plt.plot(cumulative_ret_rsi_vix_1, label='XOM-VLO (RSI + VIX Filter)')
plt.plot(cumulative_ret_rsi_vix_2, label='PG-UNP (RSI + VIX Filter)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Cumulative Returns: RSI + Volatility Regime Filter')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Portfolio returns with Current two Filters (RSI, VIX) and Risk-Based Allocations

# Compute rolling volatilities
vol_rsi_vix_1 = strategy_ret_rsi_vix_1.rolling(window=20).std().replace(0, 1e-6)
vol_rsi_vix_2 = strategy_ret_rsi_vix_2.rolling(window=20).std().replace(0, 1e-6)

# 1. Invert vol (assumes you've already replaced 0s in vol)
inv_vol_rsi_vix_1 = 1 / vol_rsi_vix_1
inv_vol_rsi_vix_2 = 1 / vol_rsi_vix_2

# 2. Clean up any lingering infinities or NaNs
inv_vol_rsi_vix_1 = inv_vol_rsi_vix_1.replace([np.inf, -np.inf], np.nan).fillna(0)
inv_vol_rsi_vix_2 = inv_vol_rsi_vix_2.replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. Continue as normal
total_inv_vol = inv_vol_rsi_vix_1 + inv_vol_rsi_vix_2
weight_rsi_vix_1 = (inv_vol_rsi_vix_1 / total_inv_vol).fillna(0)
weight_rsi_vix_2 = (inv_vol_rsi_vix_2 / total_inv_vol).fillna(0)

# Combined portfolio return
combined_ret_rsi_vix = weight_rsi_vix_1 * strategy_ret_rsi_vix_1 + weight_rsi_vix_2 * strategy_ret_rsi_vix_2
combined_cumret_rsi_vix = (1 + combined_ret_rsi_vix).cumprod()

#%%
# Combined Returns
plt.figure(figsize=(12, 6))
plt.plot(combined_cumret_rsi_vix, label='Combined (RSI + VIX Filters, Risk-Based)')
plt.axhline(1, color='gray', linestyle='--')
plt.title('Combined Portfolio Return: RSI + VIX Filters (Risk-Based Allocation)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
print("Signals active after 3/28:")
print("XOM-VLO:", test_log_1['Signal_RSI_VIX_1'].loc['2025-03-28':].value_counts())
print("PG-UNP:", test_log_2['Signal_RSI_VIX_2'].loc['2025-03-28':].value_counts())

#%%
# Convert Final Filtered Signals into Tradeable Positions

# Final filtered signal already contains the entry side (+1 for XOM, -1 for VLO)
# We assume equal and opposite positions for each leg

# XOM-VLO
test_log_1['Position_XOM'] = test_log_1['Signal_RSI_VIX_1']
test_log_1['Position_VLO'] = -test_log_1['Signal_RSI_VIX_1']

# PG-UNP
test_log_2['Position_PG'] = test_log_2['Signal_RSI_VIX_2']
test_log_2['Position_UNP'] = -test_log_2['Signal_RSI_VIX_2']

#%%
# Define Inputs for XOM-VLO

px_xom_1 = np.exp(test_log_1['XOM'])
px_vlo_1 = np.exp(test_log_1['VLO'])

positions_xom = test_log_1['Position_XOM']
positions_vlo = test_log_1['Position_VLO']

final_ret_xom = px_xom_1.pct_change()
final_ret_vlo = px_vlo_1.pct_change()

#%%
# Compute Daily PnL for XOM-VLO
pnl_xom_vlo = positions_xom.shift(1) * final_ret_xom + positions_vlo.shift(1) * final_ret_vlo
cumulative_pnl_xom_vlo = (1 + pnl_xom_vlo.fillna(0)).cumprod()

#%%
# Creating Trade Log XOM-VLO
# Combine daily positions into a full log
trade_log_xom_vlo = test_log_1[['Position_XOM', 'Position_VLO']].copy()

#%%
# Same Procees for PG-UNP

px_pg_2 = np.exp(test_log_2['PG'])
px_unp_2 = np.exp(test_log_2['UNP'])

positions_pg = test_log_2['Position_PG']
positions_unp = test_log_2['Position_UNP']

final_ret_pg = px_pg_2.pct_change()
final_ret_unp = px_unp_2.pct_change()

#%%

pnl_pg_unp = positions_pg.shift(1) * final_ret_pg + positions_unp.shift(1) * final_ret_unp
cumulative_pnl_pg_unp = (1 + pnl_pg_unp.fillna(0)).cumprod()

#%%
# Creating Trade Log PG-UNP
# Combine daily positions into a full log
trade_log_pg_unp = test_log_2[['Position_PG', 'Position_UNP']].copy()

#%%
# Creating a Combined Trade Log for Easier Implementation
# Clean and label XOM-VLO
log_xom_vlo_clean = test_log_1[['Position_XOM', 'Position_VLO']].copy()
log_xom_vlo_clean = log_xom_vlo_clean.rename(columns={'Position_XOM': 'Position_1', 'Position_VLO': 'Position_2'})
log_xom_vlo_clean['Pair'] = 'XOM-VLO'
log_xom_vlo_clean['Date'] = pd.to_datetime(log_xom_vlo_clean.index)

# Clean and label PG-UNP
log_pg_unp_clean = test_log_2[['Position_PG', 'Position_UNP']].copy()
log_pg_unp_clean = log_pg_unp_clean.rename(columns={'Position_PG': 'Position_1', 'Position_UNP': 'Position_2'})
log_pg_unp_clean['Pair'] = 'PG-UNP'
log_pg_unp_clean['Date'] = pd.to_datetime(log_pg_unp_clean.index)

# Combine both logs into final trade log
final_trade_log = pd.concat([log_xom_vlo_clean, log_pg_unp_clean], ignore_index=True)
final_trade_log.set_index('Date', inplace=True)

# Optional sanity check
print(final_trade_log['Position_1'].value_counts(dropna=False))
print(final_trade_log['Position_2'].value_counts(dropna=False))

#%% 🔍 Pivoted Trade Log for Visual Inspection (Optional)

pivot_xom_vlo = test_log_1[['Position_XOM', 'Position_VLO']].rename(columns={'Position_XOM': 'XOM', 'Position_VLO': 'VLO'})
pivot_pg_unp = test_log_2[['Position_PG', 'Position_UNP']].rename(columns={'Position_PG': 'PG', 'Position_UNP': 'UNP'})

# Create full index and reindex both
full_index = pivot_xom_vlo.index.union(pivot_pg_unp.index)
pivot_xom_vlo = pivot_xom_vlo.reindex(full_index).fillna(0)
pivot_pg_unp = pivot_pg_unp.reindex(full_index).fillna(0)

# Combine into pivot view
pivot_trade_log = pd.concat([pivot_xom_vlo, pivot_pg_unp], axis=1).fillna(0)
pivot_trade_log = pivot_trade_log[['XOM', 'VLO', 'PG', 'UNP']]

#%%
# IBKR Order Formatting
# Using Hardcoded, Fixed Dollar Exposure per pair
# Note to self, if Successful and want to implement real money, convert to dynamic order size based on vol, capital, or price

# Defining Capital per Pair
capital_per_pair = 250_000

# Per Leg
capital_per_leg = capital_per_pair / 2

#%%
# Computing Shares to Trade per day
# For XOM-VLO
shares_xom = capital_per_leg / px_xom_1
shares_vlo = capital_per_leg / px_vlo_1

# For PG-UNP
shares_pg = capital_per_leg / px_pg_2
shares_unp = capital_per_leg / px_unp_2

shares_xom = shares_xom.round().astype(int)
shares_vlo = shares_vlo.round().astype(int)
shares_pg = shares_pg.round().astype(int)
shares_unp = shares_unp.round().astype(int)

#%%

# Align final_trade_log to dates that exist in shares_xom
common_dates = final_trade_log.index
for shares_series in [shares_xom, shares_vlo, shares_pg, shares_unp]:
    common_dates = common_dates.intersection(shares_series.index)

final_trade_log = final_trade_log.loc[common_dates]

#%%
# Trading Pause/Resume Toggle 
# Using a Config File Toggle via json (toggle.json)

import json
import os

# 🔧 Define directory and file paths relative to script location
base_dir = os.path.dirname(__file__)
toggle_path = os.path.join(base_dir, 'toggle.json')
positions_file = os.path.join(base_dir, 'last_positions.json')
shares_file = os.path.join(base_dir, 'last_shares.json')
log_file = os.path.join(base_dir, 'submitted_orders_log.csv')

# Load trading toggle
with open(toggle_path, 'r') as f:
    config = json.load(f)
TRADING_ACTIVE = config.get('TRADING_ACTIVE', False)

# Load or initialize last_positions
if os.path.exists(positions_file):
    with open(positions_file, 'r') as f:
        last_positions = json.load(f)
else:
    last_positions = {}

# Load or initialize last_shares
if os.path.exists(shares_file):
    with open(shares_file, 'r') as f:
        last_shares = json.load(f)
else:
    last_shares = {}

#%%
# IBKR Order Formatting
# Formats and Stores Trade Instructions in a list without Sending anything to IBKR
# Used for previewing, simulation, dry-runs, abd exporting orders
# Only builds a structured list of orders (ticker, action, qty, pair, date) but does NOT place or log any orders
# Needs to be # out  when Live Paper Trading Starts

#formatted_orders = []

#if TRADING_ACTIVE:
#    for date, row in final_trade_log.iterrows():
#        pair = row['Pair']
#        pos1 = row['Position_1']
#        pos2 = row['Position_2']

#        if pair == 'XOM-VLO':
#            ticker_1, ticker_2 = 'XOM', 'VLO'
#            qty_1 = int(abs(shares_xom.loc[date]) if date in shares_xom.index else 0)
#            qty_2 = int(abs(shares_vlo.loc[date]) if date in shares_vlo.index else 0)
#        elif pair == 'PG-UNP':
#            ticker_1, ticker_2 = 'PG', 'UNP'
#            qty_1 = int(abs(shares_pg.loc[date]) if date in shares_pg.index else 0)
#            qty_2 = int(abs(shares_unp.loc[date]) if date in shares_unp.index else 0)
#        else:
#            continue  # skip unknown pairs

        # Only format if there's a non-zero position
#        if pos1 != 0:
#            formatted_orders.append({
#                'date': date,
#                'ticker': ticker_1,
#                'action': 'BUY' if pos1 > 0 else 'SELL',
#                'quantity': qty_1,
#                'pair': pair
#            })
#        if pos2 != 0:
#            formatted_orders.append({
#                'date': date,
#                'ticker': ticker_2,
#                'action': 'BUY' if pos2 > 0 else 'SELL',
#                'quantity': qty_2,
#                'pair': pair
#            })
#else:
#    print("Trading is PAUSED. No orders will be submitted.")
    
#%%

# Open TWS Before here
# Make Sure: 
    #  “Enable ActiveX and Socket Clients” is checked.
    #  “Read-Only API” is unchecked.
    #  Port is 7497 (for paper trading).
# Havent run from under here
#%%
# Order Formatting
# pip install ib_insync

from ib_insync import IB, Stock, MarketOrder
import sys
import nest_asyncio
nest_asyncio.apply()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)  # 7497 for paper, 7496 for live
print("Connection status:", ib.isConnected())

# Right after ib.connect(...)
@ib.errorEvent
def handle_error(reqId, errorCode, errorString, contract):
    print(f"❌ API Error [{errorCode}]: {errorString}")
#%%
# TRY BELOW PART TO SEE IF LOOPING ISSUE STOPS
# Patch for Spyder/Jupyter environments (safe to run everywhere)
#try:
#    import nest_asyncio
#    nest_asyncio.apply()
#except ImportError:
#    print("Optional: Run `pip install nest_asyncio` if using Spyder or Jupyter.")

# Initialize IB connection
#ib = IB()
#try:
#    ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)  # 7497 = paper, 7496 = live
#    print("✅ IBKR connection successful.")
#except RuntimeError as e:
#    print("⚠️ Spyder event loop conflict. Try running from CMD or install nest_asyncio.")
#    print("Error details:", e)
#    sys.exit(1)
#except Exception as e:
#    print("❌ IBKR connection failed. Check if TWS is open and API is enabled.")
#    print("Error details:", e)
#    sys.exit(1)

# Confirm status
#print("Connection status:", ib.isConnected())

#%%
# Helper Function to Create Contract and Order

def create_order(ticker, action, quantity):
    contract = Stock(ticker, 'SMART', 'USD')
    order = MarketOrder(action, quantity)
    return contract, order

#%%
# Adding Logging of Submitted Orders

import csv
from datetime import datetime

# Initialize CSV file
log_file = 'submitted_orders_log.csv'

# Write headers once if file doesn't exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Date', 'Pair', 'Ticker', 'Action', 'Quantity'])
        
def log_submitted_orders(date, pair, ticker_1, action_1, qty_1, ticker_2, action_2, qty_2):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Leg 1
        writer.writerow([timestamp, date, pair, ticker_1, action_1, qty_1])
        # Leg 2
        writer.writerow([timestamp, date, pair, ticker_2, action_2, qty_2])

def log_trade(date, pair, ticker, action, quantity):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, date, pair, ticker, action, quantity])
        
#%%
# Make sure final_trade_log has datetime index
final_trade_log.index = pd.to_datetime(final_trade_log.index)

# Get only the most recent trading day's signals
latest_date = final_trade_log.index.max()
today_trades = final_trade_log.loc[final_trade_log.index == latest_date]

#%%
print("🧩 Confirming last_positions just before placing orders:")
print(json.dumps(last_positions, indent=2))

print("📦 Confirming last_shares just before placing orders:")
print(json.dumps(last_shares, indent=2))

#%%
# Loop Through final_trade_log and Generate Orders
# Creates contracts
# Places Orders to IBKR
# Logs orders to CSV

if TRADING_ACTIVE:
    print(f"\n🚀 LIVE TRADING MODE — Executing trades for {latest_date.date()}")

    print("🧩 Confirming last_positions just before placing orders:")
    print(json.dumps(last_positions, indent=2))

    print("📦 Confirming last_shares just before placing orders:")
    print(json.dumps(last_shares, indent=2))

    for date, row in today_trades.iterrows():
        pair = row['Pair']
        signal_1 = row['Position_1']
        signal_2 = row['Position_2']

        if pair == 'XOM-VLO':
            ticker_1, ticker_2 = 'XOM', 'VLO'
            qty_1_default = int(abs(shares_xom.get(date, 0)))
            qty_2_default = int(abs(shares_vlo.get(date, 0)))
        elif pair == 'PG-UNP':
            ticker_1, ticker_2 = 'PG', 'UNP'
            qty_1_default = int(abs(shares_pg.get(date, 0)))
            qty_2_default = int(abs(shares_unp.get(date, 0)))
        else:
            print(f"❌ Unknown pair: {pair}")
            continue

        key_1 = f"{pair}_1"
        key_2 = f"{pair}_2"
        last_pos_1 = last_positions.get(key_1, 0)
        last_pos_2 = last_positions.get(key_2, 0)

        signal_changed = (signal_1 != last_pos_1) or (signal_2 != last_pos_2)

        if not signal_changed:
            print(f"✅ Skipping {pair} — no signal change detected")
            continue

        qty_1 = int(abs(last_shares.get(key_1, qty_1_default))) if signal_1 == 0 else qty_1_default
        qty_2 = int(abs(last_shares.get(key_2, qty_2_default))) if signal_2 == 0 else qty_2_default

        action_1 = 'BUY' if signal_1 > 0 else 'SELL' if signal_1 < 0 else None
        action_2 = 'BUY' if signal_2 > 0 else 'SELL' if signal_2 < 0 else None

        try:
            if signal_1 != 0 and qty_1 > 0:
                contract_1 = Stock(ticker_1, 'SMART', 'USD')
                ib.qualifyContracts(contract_1)
                order_1 = MarketOrder(action_1, qty_1)
                ib.placeOrder(contract_1, order_1)
                print(f"📈 Placed {action_1} order for {qty_1} shares of {ticker_1}")
            elif signal_1 == 0 and qty_1 > 0:
                contract_1 = Stock(ticker_1, 'SMART', 'USD')
                ib.qualifyContracts(contract_1)
                close_action_1 = 'SELL' if last_pos_1 > 0 else 'BUY'
                order_1 = MarketOrder(close_action_1, qty_1)
                ib.placeOrder(contract_1, order_1)
                print(f"📉 Closed {qty_1} shares of {ticker_1}")

            if signal_2 != 0 and qty_2 > 0:
                contract_2 = Stock(ticker_2, 'SMART', 'USD')
                ib.qualifyContracts(contract_2)
                order_2 = MarketOrder(action_2, qty_2)
                ib.placeOrder(contract_2, order_2)
                print(f"📈 Placed {action_2} order for {qty_2} shares of {ticker_2}")
            elif signal_2 == 0 and qty_2 > 0:
                contract_2 = Stock(ticker_2, 'SMART', 'USD')
                ib.qualifyContracts(contract_2)
                close_action_2 = 'SELL' if last_pos_2 > 0 else 'BUY'
                order_2 = MarketOrder(close_action_2, qty_2)
                ib.placeOrder(contract_2, order_2)
                print(f"📉 Closed {qty_2} shares of {ticker_2}")

            # Save signal and shares after successful order
            last_positions[key_1] = signal_1
            last_positions[key_2] = signal_2
            with open('last_positions.json', 'w') as f:
                json.dump(last_positions, f, indent=2)

            last_shares[key_1] = qty_1
            last_shares[key_2] = qty_2
            with open('last_shares.json', 'w') as f:
                json.dump(last_shares, f, indent=2)

            # Log the trade
            log_trade(latest_date, pair, ticker_1, action_1 or close_action_1, qty_1)
            log_trade(latest_date, pair, ticker_2, action_2 or close_action_2, qty_2)

        except Exception as e:
            print(f"❌ Order failed for {pair}: {e}")
else:
    print("🚫 Trading is PAUSED. No orders will be submitted.")


#%%

with open(positions_file, 'w') as f:
    json.dump(last_positions, f, indent=2)

with open(shares_file, 'w') as f:
    json.dump(last_shares, f, indent=2)

#%%
# Print Checks before Trading
#%%
# Confirm Trading Toggle
print(f"TRADING_ACTIVE: {TRADING_ACTIVE}")
#%%
# Confirm API Connection
print(f"Connection status: {ib.isConnected()}")
#%%
# Confirm Price Data Loaded Properly
print("Price data preview:")
print(price_data.head())

#%%
print(shares_xom.head())
print(shares_vlo.head())
print(final_trade_log.head())

#%%
print(final_trade_log.index.difference(shares_xom.index))
#%%
for date, row in final_trade_log.iterrows():
    in_index = date in shares_xom.index
    print(f"Date: {date}, In shares_xom: {in_index}")

#%%
print("final_trade_log index sample:")
print(final_trade_log.index[:5])
print("shares_xom index sample:")
print(shares_xom.index[:5])
#%%
print("final_trade_log index type:", type(final_trade_log.index[0]))
print("shares_xom index type:", type(shares_xom.index[0]))

#%%
print("final_trade_log index head:\n", final_trade_log.index[:5])
print("final_trade_log sample:\n", final_trade_log.head())
#%%



#%%
print("➡️  Final Trade Log Date Range:", final_trade_log.index.min(), "to", final_trade_log.index.max())
print("➡️  shares_xom Date Range:", shares_xom.index.min(), "to", shares_xom.index.max())
#%%
print("✅ Common dates count:", len(final_trade_log.index))
print("🧪 Sample dates from final_trade_log after filtering:", final_trade_log.index[:5])

#%%
sample_date = final_trade_log.index[0]
print("📅 Sample date:", sample_date)
print("🔍 In shares_xom:", sample_date in shares_xom.index)

#%%
for date, row in final_trade_log.head(5).iterrows():
    qty_check = shares_xom.get(date, None)
    print(f"📆 {date} | Signal: {row['Position_1']} | Qty XOM: {qty_check}")
#%%


# Final Checks

#%%
print("📂 Loaded previous positions:", last_positions)
#%%
print("💾 Updated positions saved:", last_positions)
#%%
if today_trades.empty:
    print("⚠️ No trades to place today. No signal change or no new signals.")

#%%
print(f"✅ Order log updated for {latest_date.date()}")

#%%
# Optional enhancement
if latest_date.weekday() >= 5:
    print("⛔ It's a weekend. No trading.")
    exit()

#%%

#%%

print(f"\n🔎 Evaluating pair: {pair}")
print(f"Signal_1: {signal_1}, Signal_2: {signal_2}")
print(f"Qty_1: {qty_1}, Qty_2: {qty_2}")
print(f"Action_1: {action_1}, Action_2: {action_2}")
print(f"Last positions: {last_positions.get(f'{pair}_1')}, {last_positions.get(f'{pair}_2')}")
print(f"Live flag: {TRADING_ACTIVE}, Trade condition 1: {signal_1 != last_positions.get(f'{pair}_1')}, Trade condition 2: {signal_2 != last_positions.get(f'{pair}_2')}")
#%%
print(date in shares_xom.index)
print(date in shares_pg.index)
print(date in shares_unp.index)
print(shares_pg.get(date, "Missing"))
print(shares_unp.get(date, "Missing"))
#%%
print(f"\n📅 Latest trade date: {latest_date.date()}")

for pair, shares_series_1, shares_series_2 in [
    ('XOM-VLO', shares_xom, shares_vlo),
    ('PG-UNP', shares_pg, shares_unp)
]:
    qty_1 = shares_series_1.get(latest_date, 'Missing')
    qty_2 = shares_series_2.get(latest_date, 'Missing')
    print(f"🔍 {pair} | Qty_1: {qty_1}, Qty_2: {qty_2}")

#%%
# DRY TEST RUN TO VALIDATE LOGIC
# ✅ Toggle this to True when testing logic, False for live runs
DRY_RUN = False

if DRY_RUN:
    print("\n🧪 DRY RUN — Trade Preview for", latest_date.date())

    print("🧩 Confirming last_positions:")
    print(json.dumps(last_positions, indent=2))
    print("📦 Confirming last_shares:")
    print(json.dumps(last_shares, indent=2))

    for date, row in today_trades.iterrows():
        pair = row['Pair']
        signal_1 = row['Position_1']
        signal_2 = row['Position_2']

        if pair == 'XOM-VLO':
            ticker_1, ticker_2 = 'XOM', 'VLO'
            qty_1_default = int(abs(shares_xom.get(date, 0)))
            qty_2_default = int(abs(shares_vlo.get(date, 0)))
        elif pair == 'PG-UNP':
            ticker_1, ticker_2 = 'PG', 'UNP'
            qty_1_default = int(abs(shares_pg.get(date, 0)))
            qty_2_default = int(abs(shares_unp.get(date, 0)))
        else:
            print(f"❌ Unknown pair: {pair}")
            continue

        key_1 = f"{pair}_1"
        key_2 = f"{pair}_2"
        last_position_1 = last_positions.get(key_1, 0)
        last_position_2 = last_positions.get(key_2, 0)

        qty_1 = int(abs(last_shares.get(key_1, qty_1_default))) if signal_1 == 0 else qty_1_default
        qty_2 = int(abs(last_shares.get(key_2, qty_2_default))) if signal_2 == 0 else qty_2_default

        action_1 = 'BUY' if signal_1 > 0 else 'SELL' if signal_1 < 0 else None
        action_2 = 'BUY' if signal_2 > 0 else 'SELL' if signal_2 < 0 else None

        should_trade = signal_1 != last_position_1 or signal_2 != last_position_2

        print(f"\n🔍 Pair: {pair}")
        print(f"  Signal_1: {signal_1}, Last: {last_position_1} → Qty: {qty_1} → Action: {action_1}")
        print(f"  Signal_2: {signal_2}, Last: {last_position_2} → Qty: {qty_2} → Action: {action_2}")
        print(f"  Should Trade: {should_trade}")

        if should_trade:
            if signal_1 != 0 and qty_1 > 0:
                print(f"  ✅ WOULD PLACE: {action_1} {qty_1} of {ticker_1}")
            elif signal_1 == 0 and qty_1 > 0:
                close_action_1 = 'SELL' if last_position_1 > 0 else 'BUY'
                print(f"  🔁 WOULD CLOSE: {close_action_1} {qty_1} of {ticker_1}")

            if signal_2 != 0 and qty_2 > 0:
                print(f"  ✅ WOULD PLACE: {action_2} {qty_2} of {ticker_2}")
            elif signal_2 == 0 and qty_2 > 0:
                close_action_2 = 'SELL' if last_position_2 > 0 else 'BUY'
                print(f"  🔁 WOULD CLOSE: {close_action_2} {qty_2} of {ticker_2}")
        else:
            print("  🚫 No signal change — no orders would be submitted.")

#%%
# ✅ Toggle to True to test contract qualification
TEST_CONTRACTS = False

if TEST_CONTRACTS:
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("⚠️ nest_asyncio not found. Install with: pip install nest_asyncio")

    for ticker in ['XOM', 'VLO', 'PG', 'UNP']:
        try:
            contract = Stock(ticker, 'SMART', 'USD')
            result = ib.qualifyContracts(contract)

            if result:
                print(f"✅ {ticker} qualified: {result[0].conId}, exchange: {result[0].exchange}, primary: {result[0].primaryExchange}")
            else:
                print(f"❌ {ticker} not qualified — possible issue.")
        except Exception as e:
            print(f"❌ {ticker} error: {e}")

#%%
print("Final Trade Log — 6/13")
print(final_trade_log.loc["2025-06-13"])

print("\nPosition Series — 6/13")
print("XOM:", positions_xom.loc["2025-06-13"])
print("VLO:", positions_vlo.loc["2025-06-13"])
print("PG:", positions_pg.loc["2025-06-13"])
print("UNP:", positions_unp.loc["2025-06-13"])

#%%
print("📂 Current working directory:", os.getcwd())
#%%