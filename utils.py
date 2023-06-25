import multiprocessing
from typing import Any, Callable, List, Optional
import numpy as np
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

from functools import wraps
from time import perf_counter
from typing import Callable
from typing import Tuple

def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        return results, run_time
    return wrapper

@timer
def p_map(items: List[Any], func: Callable, options: Optional[dict[str, any]] = None) -> Any:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, items))
    return results

def compose(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def filter_dates(start: str, end: str, df: pd.DataFrame) -> pd.DataFrame:
    if start and end:
        df = df.loc[start:end]
    return df

def filter_symbols(symbols: List[str], trades: pd.DataFrame) -> pd.DataFrame:
    # filter all rows where column symbol is not in symbols list:
    filtered_trades = trades[trades['symbol'].isin(symbols)]
    return filtered_trades

def filter_start_date(date: str, df: pd.DataFrame) -> pd.DataFrame:
    starting_date = pd.to_datetime(date)
    return df[df.index >= starting_date]


def filter_daily_rank(metric: str, trades: pd.DataFrame) -> pd.DataFrame:
    def process_group(group):
        date = group.name
        print(f"{date}: sorting on rank and removing")                          
    grouped = trades.groupby(trades.index.date)
    grouped.apply(process_group)


def apply_daily_rank(metric: str, trades: pd.DataFrame, tickers_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def apply_rank(row):        
        ticker_df = tickers_dict[row['symbol']]
        start_date = row.name
        row[metric] = ticker_df.loc[start_date][metric]
        #print(f"{start_date}: added {ticker_df.loc[start_date][metric]} for {row['symbol']}")        
        return row     
    return trades.apply(apply_rank, axis=1)


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

def parse_csv_to_dataframe(symbol: str, path) -> pd.DataFrame:
    file_path = f"{path}/{symbol}.csv"
    data = pd.read_csv(file_path)
    
    # Parse time column as DateTimeIndex
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)   
    data.name = symbol
    
    return data

def load_tickers(path: str, tickers=List[str]) -> List[pd.DataFrame]:    
    data = []
    for ticker in tickers:
        df = json_to_df(symbol=ticker, path=f"{path}/{ticker}.json")
        data.append(df)
    return data


def load_json_data(symbol: str, path: str) -> Optional[dict]:
    with open(path) as f:
        json_data = json.load(f)
        return json_data.get(symbol)

def process_json_data(data: dict, symbol: str) -> pd.DataFrame:
    if data:
        df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.set_index('DateTime', inplace=True)
        # TODO: add freq?
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York').tz_localize(None)
        #TODO: remove extended hours?
        df.name = symbol
        return df
    else:
        print(f"Symbol {symbol} not found in the json")

def json_to_df(symbol: str, path: str) -> Optional[pd.DataFrame]:
    data = load_json_data(symbol, path)
    if data is not None:
        return process_json_data(data, symbol)

    return None

@timer
def load_json_to_df_async(root_path: str, symbols: list[str]) -> list[Optional[pd.DataFrame]]:
    async def process_file(symbol, path):
        print(f"Processing file: {path}")
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, load_json_data, symbol, path)
        return process_json_data(data, symbol)

    async def load_files():
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(process_file(symbol, f"{root_path}/{symbol}.json"))
            tasks.append(task)
        return await asyncio.gather(*tasks)

    return asyncio.run(load_files())

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