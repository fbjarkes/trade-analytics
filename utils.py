from typing import List
import pandas as pd
import json

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

def apply_daily_rank(metric: str, trades: pd.DataFrame) -> pd.DataFrame:
    grouped = trades.groupby(trades.index.date)
    for date, group in grouped:
        print(f"Processing date: {date}")
        print(group)
    return trades
    

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

def json_to_df(symbol: str, path: str) -> pd.DataFrame:  
    print(f"Reading json file: {path}")  
    with open(path) as f:
        json_data = json.load(f)
        if json_data[symbol]:
            df = pd.DataFrame(json_data[symbol], columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('DateTime', inplace=True)
            df.index = pd.DatetimeIndex(df.index) # TODO: add freq?
            df.name = symbol
            return df
        else:
            print(f"Symbol {symbol} not found in json '{path}'")

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