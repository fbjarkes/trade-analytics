from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats
from finta import TA

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

RANK_METRICS = [
    #'EMA100_DISTANCE',
    'PRICE',
    'RSI5',
    #'RSI10',
    #'RSI20',
]

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

def apply_rank_metric(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    for metric in metrics:
        if metric == 'EMA100_DISTANCE':
            df[metric] = (df['Close'] - TA.EMA(df, 100)) / TA.ATR(df, 50)
        if metric == 'PRICE':
            df[metric] = df['Close']
        if metric == 'RSI5':
            df[metric] = TA.RSI(df, 5)
        if metric == 'RSI10':
            df[metric] = TA.RSI(df, 10)
        if metric == 'RSI20':
            df[metric] = TA.RSI(df, 20)
    return df

def apply_rank_metric_multi(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(apply_rank_metric, dfs))
    return results
