from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from finta import TA
import streamlit as st

import pandas as pd


from typing import Any, Dict, List, Tuple

RANK_METRICS = [
    #'EMA100_DISTANCE',
    'PRICE',
    'RSI5',
    #'RSI10',
    #'RSI20',
]
SELECTABLE_METRICS = ['ALL', 'RANDOM'] + [f"{metric}_ASC" for metric in RANK_METRICS] + [f"{metric}_DESC" for metric in RANK_METRICS]
EC_FILTER_METRICS = ['NONE', 'ABOVE_MA_5', 'ABOVE_MA_10', 'ABOVE_MA_20']
INDEX_METRICS_NORMALIZED = ['EMA_10_20_ABOVE', 'EMA_10_20_BELOW', '10_BELOW', '10_ABOVE']
INDEX_METRICS = ['EMA_10_ABOVE_20', 'EMA_10_BELOW_20', 'EMA_5_CLOSE_ABOVE', 'EMA_10_CLOSE_ABOVE', 'EMA_20_CLOSE_ABOVE', 'EMA_50_CLOSE_ABOVE', 'ABOVE_VALUE', 'BELOW_VALUE']


def max_open(trades: pd.DataFrame) -> int:
    max_in_progress = 0
    for timestamp in trades.index.unique():
        in_progress = len(trades[(timestamp >= trades.index) & (timestamp <= trades['end_date'])])
        max_in_progress = max(max_in_progress, in_progress)
    return max_in_progress


def get_trade_stats(df: pd.DataFrame, start_eq: float) -> Tuple[Dict[str, Any], pd.Series]:
    if df.empty:
        return {'Mean': 0, 'Std': 0, 'Max Open': 0, 'Return (%)': 0, 'Max DD (%)': 0, 'Max Exposure': 0}, pd.Series()
    mean = df["pnl"].mean()
    std = df["pnl"].std()
    max_trades = max_open(df)
    if len(df) > 0:
        pnl = df.sort_values('end_date')['pnl']
        cum_sum = pnl.cumsum()
        ret = cum_sum[-1]/start_eq*100
        cum_max = pnl.cummax()
        dd = pnl - cum_max
        dd_pct = dd / cum_max * 100
        max_drawdown = dd_pct.min()
    else:
        cum_sum = pd.Series()
        ret = 0
        max_drawdown = 0
    return {'Mean': mean, 'Std': std, 'Max Open': max_trades, 'Return (%)': ret, 'Max DD (%)': max_drawdown,
            'Max Exposure': 0}, cum_sum


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


def apply_rank(metrics: List[str], trades: pd.DataFrame, tickers_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def apply_group_rank(group):
        daily = group.name[0]
        m15 = group.name[1]
        for metric in metrics:
            group[f"{metric}_ASC"] = group[metric].rank(method='min', ascending=True)
            group[f"{metric}_DESC"] = group[metric].rank(method='min', ascending=False)
        return group

    def apply_metrics(row):
        ticker_df = tickers_dict[row['symbol']]
        start_date = row.name
        
        for metric in metrics:
            row[metric] = ticker_df.loc[start_date][metric]
            #row.loc[metric] = ticker_df.loc[start_date][metric]
        #print(f"{start_date}: added {ticker_df.loc[start_date][metric]} for {row['symbol']}")        
        return row

    start = perf_counter()
    trades = trades.apply(apply_metrics, axis=1)
    print(f"Apply metrics: {perf_counter() - start:.2f} seconds")
    start = perf_counter()
    groups = trades.groupby([pd.Grouper(freq='D'), pd.Grouper(freq='15Min')])
    print(f"Group by: {perf_counter() - start:.2f} seconds")
    start = perf_counter()
    processed_groups = groups.apply(apply_group_rank).droplevel(0).droplevel(1)
    print(f"Apply group rank: {perf_counter() - start:.2f} seconds")
    return processed_groups


def filter_rank(trades: pd.DataFrame, metric: str, rank: int) -> pd.DataFrame:
    if metric == 'ALL' or metric is None:
        return trades
    elif metric == 'RANDOM':
        if st.session_state.timeframe != 'day':
            st.error(f"Random metric not implemented for tf {st.session_state.timeframe}")
            return trades
        def select_random(group):
            if not group.empty:
                return group.sample(n=min(rank, len(group)))
            else:
                return group
        groups = trades.groupby([pd.Grouper(freq='D')])
        trades = groups.apply(select_random).droplevel(0)
        return trades
    else:
        try:
            return trades.loc[trades[f"{metric}"] <= rank]
        except KeyError as e:
            st.error(f"Missing rank '{metric}' in trades")
            return trades


def filter_start_date(trades: pd.DataFrame, date: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    starting_date = pd.to_datetime(date)
    return trades[trades.index >= starting_date]


def filter_symbols(trades: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if symbols is None or (len(symbols) == 1 and symbols[0] == ''):
        return trades
    filtered_trades = trades[trades['symbol'].isin(symbols)]
    return filtered_trades


def filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if start and end:
        df = df.loc[start:end]
    return df


