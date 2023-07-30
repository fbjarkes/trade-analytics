

from finta import TA
from typing import List
import pandas as pd

import streamlit as st

class IndexFilter:
    
    def filter_by_index(index: pd.DataFrame, index_metric: str, trades: pd.DataFrame, tf: str, value=10) -> pd.DataFrame:
        pass




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
