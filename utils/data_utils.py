from functools import partial
import json
from utils import stats_utils
from utils.func_utils import timer, p_map

import pandas as pd


import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

#TODO: should not have anything with Streamlit here
import streamlit as st

# def json_to_df(symbol: str, path: str) -> pd.DataFrame:  
#     print(f"Reading json file: {path}")  
#     with open(path) as f:
#         json_data = json.load(f)
#         if json_data[symbol]:
#             df = pd.DataFrame(json_data[symbol], columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
#             df.set_index('DateTime', inplace=True)
#             df.index = pd.DatetimeIndex(df.index) # TODO: add freq?
#             df.name = symbol
#             return df
#         else:
#             print(f"Symbol {symbol} not found in json '{path}'")

# TODO: to plot index with plotly?
def parse_uploaded_csv(file):
    print(f"Parsing csv {file}")
    data = pd.read_csv(file)    
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data = data.resample('D').ffill() # TODO: best way to remove the time part of DateTimeIndex?    
    name = file.name.split('.')[0].upper()
    return data, name


def process_json_data(data: dict, symbol: str) -> pd.DataFrame:
    if data:        
        df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.set_index('DateTime', inplace=True)
        # TODO: add freq?
        # Convert to Wall Street time since all trades have start/dates in Wall Street time
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York').tz_localize(None)
        #TODO: remove extended hours?
        df.name = symbol
        return df
    else:
        print(f"Symbol {symbol} not found in the json")
        
def process_csv_data(path: str, symbol) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col='time', parse_dates=False, usecols=['time', 'open', 'high', 'low', 'close', 'Volume'])
        # Convert to Wall Street time since all trades have start/dates in Wall Street time
        df.index = pd.to_datetime(df.index, unit='s', utc=True).tz_convert('America/New_York').tz_localize(None)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)      
        df.name = symbol
        print(f"{len(df)} rows for {symbol}")
        return df
    except Exception as e:
        print(f"Error parsing csv '{path}': {e}")

def load_json_data(symbol: str, path: str) -> Optional[Dict]:
    with open(path) as f:
        json_data = json.load(f)       
        return json_data.get(symbol)

#TODO: does this actually work
def load_json_to_df_threads(symbols: list[str], paths: list[str]) -> list[Optional[pd.DataFrame]]:
    async def process_file(symbol, path):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, load_json_data, symbol, path)        
        return process_json_data(data, symbol)

    num_cpus = multiprocessing.cpu_count()
    results = []
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        for symbol, path in zip(symbols, paths):
            task = loop.create_task(process_file(symbol, path))
            tasks.append(task)

        results = loop.run_until_complete(asyncio.gather(*tasks))

    return results


@timer
def load_json_to_df_async(root_path: str, symbols: list[str]) -> list[Optional[pd.DataFrame]]:
    async def process_file(symbol, path):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, load_json_data, symbol, path)
        df = process_json_data(data, symbol)
        print(f"Processed file '{path}'")
        return df

    async def load_files():
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(process_file(symbol, f"{root_path}/{symbol}.json"))
            tasks.append(task)
        return await asyncio.gather(*tasks)

    return asyncio.run(load_files())

@timer
def load_tv_csv_to_df(root_path: str, symbols: list[str]) -> List[Optional[pd.DataFrame]]:
    dfs = []
    for symbol in symbols:
        df = process_csv_data(f"{root_path}/{symbol}.csv", symbol)
        dfs.append(df)
    return dfs

def json_to_df(symbol: str, path: str) -> Optional[pd.DataFrame]:
    data = load_json_data(symbol, path)
    if data is not None:
        return process_json_data(data, symbol)

    return None


def load_tickers(path: str, tickers=List[str]) -> List[pd.DataFrame]:
    data = []
    for ticker in tickers:
        df = json_to_df(symbol=ticker, path=f"{path}/{ticker}.json")
        data.append(df)
    return data


def parse_csv_to_dataframe(symbol: str, path) -> pd.DataFrame:
    file_path = f"{path}/{symbol}.csv"
    data = pd.read_csv(file_path)

    # Parse time column as DateTimeIndex
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data.name = symbol

    return data


def parse_json_to_dataframe(symbol: str, path: str) -> pd.DataFrame:
    """
    Alpaca json data
    """
    file_path = f"{path}/{symbol}.json"

    with open(file_path, 'r') as file:
        data = json.load(file)

    records = data[symbol]

    df_data = []
    for record in records:
        dt = pd.to_datetime(record['DateTime'])
        #dt = dt.tz_localize('UTC').tz_convert('America/New_York')
        df_data.append([
            dt,
            record['Open'],
            record['High'],
            record['Low'],
            record['Close'],
            record['Volume']
        ])

    df = pd.DataFrame(df_data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('DateTime', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.symbol = symbol

    return df


@st.cache_data 
def load_trades(csv_path: str) -> pd.DataFrame:
    print(f"Reading trades csv: {csv_path}")
    # Note: datetimes are in America/New_York timezone (not tz aware)
    # Columns: ts, symbol, start_date, end_date, pnl, value
    trades_df = pd.read_csv(csv_path)    
    trades_df['time'] = pd.to_datetime(trades_df['ts'], unit='s')
    trades_df['start_date'] = pd.to_datetime(trades_df['start_date'])
    trades_df['end_date'] = pd.to_datetime(trades_df['end_date'])
    trades_df.set_index('start_date', inplace=True)
    
    # sort on start_date
    trades_df.sort_index(inplace=True)
    print(f"Read {len(trades_df)} trades from '{csv_path}'")
    return trades_df


def load_tickers_data(trades: pd.DataFrame, data_path: str, provider: str) -> List[pd.DataFrame]:
    tickers = trades['symbol'].unique()
      
    ### DEBUG
    # 100 random tickers
    #tickers = np.random.choice(tickers, 100, replace=False)
    #print(f"Randomly selected {len(tickers)} tickers:", tickers)
    # filter trades by tickers
    #trades = trades[trades['symbol'].isin(tickers)]
    ### 
    try: 
        print(f"Loading data from '{data_path}' for {len(tickers)} tickers (provider={provider})")
        if provider == 'alpaca':            
            tickers_data, runtime = load_json_to_df_async(data_path, tickers)      
        elif provider == 'tv':
            tickers_data, runtime = load_tv_csv_to_df(data_path, tickers)
        else:
            raise Exception(f"Unknown provider: {provider}")
        print(f"tickers data loaded in {runtime:.2f} seconds")
        
        return tickers_data
    except Exception as e:
        import traceback
        traceback.print_exc()            
        print(f"Error loading data: {e}")            
        return []
   
   
def apply_metrics(trades: pd.DataFrame, tickers_data: List[pd.DataFrame]) -> List[pd.DataFrame]:
    try:
        # Add TA etc. to all tickers data available
        print(f"Applying metrics to tickers data ({len(tickers_data)} dataframes)")
        tickers_data, runtime = p_map(tickers_data, partial(stats_utils.apply_rank_metric, metrics=stats_utils.RANK_METRICS))
        print()
        print(f"p_map: {runtime:.2f} seconds")
        sum = 0   
        for df in tickers_data:
            sum += df.memory_usage().sum()
        print(f"Total memory usage: {sum/1000/1000:.1f} MB for {len(tickers_data)} tickers") 
        
        tickers_dict = {df.name: df for df in tickers_data} # Need symbol mapping for rank function
        print(f"Applying metrics to trades data ({len(trades)})")
        trades = stats_utils.apply_rank(stats_utils.RANK_METRICS, trades, tickers_dict)
        return trades
    except Exception as e:
        import traceback
        traceback.print_exc()            
        print(f"Error applying metrics: {e}")            
        return trades