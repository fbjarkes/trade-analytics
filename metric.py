import pandas as pd
import numpy as np
import scipy.stats as stats
from finta import TA

def ema_cloud_10_20(trades: pd.DataFrame, index: pd.DataFrame):
    merged_df = pd.merge_asof(trades, index['EMA_10_20_Cloud'], left_index=True, right_on='time', direction='nearest')
    pos_trades = merged_df.loc[merged_df['EMA_10_20_Cloud'] == True, 'pnl']
    neg_trades = merged_df.loc[merged_df['EMA_10_20_Cloud'] == False, 'pnl']
    pos_t_statistic, pos_p_value = stats.ttest_ind(trades['pnl'], pos_trades)
    neg_t_statistic, neg_p_value = stats.ttest_ind(trades['pnl'], neg_trades)
    return {
        'pos_avg': pos_trades.mean(),
        'pos_std': pos_trades.std(),
        'neg_avg': neg_trades.mean(),
        'neg_std': neg_trades.std(),
        'pos_t_statistic': pos_t_statistic,
        'pos_p_value': pos_p_value,
        'neg_t_statistic': neg_t_statistic,
        'neg_p_value': neg_p_value
    }

def ema_cross_5():
    pass

def ema_cross_10():
    pass

def ema_cross_20():
    pass

def ema_cross_50():
    pass    

metrics_functions = {
    'EMA_CLOUD_10_20': ema_cloud_10_20,
    'EMA5_CROSS_5': ema_cross_5,
    'EMA5_CROSS_10': ema_cross_10,
    'EMA5_CROSS_20': ema_cross_20,
    'EMA5_CROSS_50': ema_cross_50 
}

def index_analysis(metric: str, index: pd.DataFrame, trades: pd.DataFrame):
    func = metrics_functions[metric]
    return func(trades, index)

    