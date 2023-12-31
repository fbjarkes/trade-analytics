from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats
from finta import TA

from utils.stats_utils import apply_rank_metric

@dataclass
class TradeData:
    metric: str
    pos_count: int
    pos_avg: float
    pos_std: float
    neg_avg: float
    neg_std: float
    neg_count: int
    pos_t_statistic: float
    pos_p_value: float
    neg_t_statistic: float
    neg_p_value: float


def ema_cross_10_20(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df['EMA10'] = TA.EMA(df, 10)
    df['EMA20'] = TA.EMA(df, 20)
    df[metric] = df['EMA10'] >= df['EMA20']
    return df

def close_ema_5(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df['EMA5'] = TA.EMA(df, 5)
    df[metric] = df['close'] >= df['EMA5']
    return df

def close_ema_10(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df['EMA10'] = TA.EMA(df, 10)
    df[metric] = df['close'] >= df['EMA10']
    return df

def close_ema_20(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df['EMA20'] = TA.EMA(df, 20)
    df[metric] = df['close'] >= df['EMA20']
    return df

def close_ema_50(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df['EMA50'] = TA.EMA(df, 50)
    df[metric] = df['close'] >= df['EMA50']
    return df
    
INDEX_METRIC_FUNCTIONS = {
    'EMA_CROSS_10_20': ema_cross_10_20,
    'CLOSE_EMA_5': close_ema_5,
    'CLOSE_EMA_10': close_ema_10,
    'CLOSE_EMA_20': close_ema_20,
    'CLOSE_EMA_50':  close_ema_50
}

def index_analysis(metric: str, index: pd.DataFrame, trades: pd.DataFrame) -> Tuple[TradeData, pd.DataFrame]:
    func = INDEX_METRIC_FUNCTIONS[metric]
    func(df=index, metric=metric)
    merged_df = pd.merge_asof(trades, index[metric], left_index=True, right_on='time', direction='nearest')
    pos_trades = merged_df.loc[merged_df[metric] == True, 'pnl']
    neg_trades = merged_df.loc[merged_df[metric] == False, 'pnl']
    pos_t_statistic, pos_p_value = stats.ttest_ind(trades['pnl'], pos_trades)
    neg_t_statistic, neg_p_value = stats.ttest_ind(trades['pnl'], neg_trades)
    res = TradeData(**{
        'metric': metric,
        'pos_count': len(pos_trades),
        'pos_avg': pos_trades.mean(),
        'pos_std': pos_trades.std(),
        'neg_avg': neg_trades.mean(),
        'neg_std': neg_trades.std(),
        'neg_count': len(neg_trades),
        'pos_t_statistic': pos_t_statistic,
        'pos_p_value': pos_p_value,
        'neg_t_statistic': neg_t_statistic,
        'neg_p_value': neg_p_value
    })
    return res, merged_df


