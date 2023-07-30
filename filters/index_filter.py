

from finta import TA
from typing import List
import pandas as pd

import streamlit as st


INDEX_METRICS = ['NONE', 'EMA_10_20_ABOVE', 'EMA_10_20_BELOW', 
                 'CLOSE_ABOVE_EMA_3', 'CLOSE_ABOVE_EMA_5', 'CLOSE_ABOVE_EMA_10', 'CLOSE_ABOVE_EMA_20', 'CLOSE_ABOVE_EMA_50', 'CLOSE_ABOVE_EMA_100', 'CLOSE_ABOVE_EMA_200',
                 'CLOSE_BELOW_EMA_3', 'CLOSE_BELOW_EMA_5', 'CLOSE_BELOW_EMA_10', 'CLOSE_BELOW_EMA_20', 'CLOSE_BELOW_EMA_50', 'CLOSE_BELOW_EMA_100', 'CLOSE_BELOW_EMA_200', 
                 'CLOSE_ABOVE_VALUE', 'CLOSE_BELOW_VALUE']

INDEX = ['NONE', 'MMTW_20', 'MMFI_50', 'MMOH_100', 'MMOF_150', 'MMTH_200','OMXS30', 'IWM']


@st.cache_data
def apply_index_metric(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    print(f"Adding Index metrics")
    df['EMA3'] = TA.EMA(df, 3)
    df['EMA5'] = TA.EMA(df, 5)
    df['EMA10'] = TA.EMA(df, 10)
    df['EMA20'] = TA.EMA(df, 20)
    df['EMA50'] = TA.EMA(df, 50)
    df['EMA100'] = TA.EMA(df, 100)
    df['EMA200'] = TA.EMA(df, 200)
    
    return df

@st.cache_data
def filter_by_index(index: pd.DataFrame, index_metric: str, trades: pd.DataFrame, tf: str, value=10) -> pd.DataFrame:
    if index_metric == 'NONE':
        print("No index metric selected")
        return trades
    print(f"Adding '{index_metric}' to Index (value={value}) and filtering trades")
    if index_metric.startswith('CLOSE_ABOVE_EMA_'):
        period = int(index_metric.split('_')[-1])
        index[index_metric] = index['Close'] > index[f"EMA{period}"]
    elif index_metric.startswith('CLOSE_BELOW_EMA_'):
        period = int(index_metric.split('_')[-1])
        index[index_metric] = index['Close'] < index[f"EMA{period}"]
    elif index_metric == 'CLOSE_ABOVE_VALUE':
        index[index_metric] = index['Close'] > value
    elif index_metric == 'CLOSE_BELOW_VALUE':
        index[index_metric] = index['Close'] < value
    else:
        print(f"{index_metric} not implemented")
        return trades

    #TODO: works for intraday timeframes (assuming Index is daily) ?
    t = pd.merge_asof(trades, index[index_metric], left_index=True, right_on='time', direction='nearest')

    # filter trades
    t = t[t[index_metric] == True]
    return t

