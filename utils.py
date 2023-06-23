import pandas as pd
import json


def filter_dates(start: str, end: str, df: pd.DataFrame) -> pd.DataFrame:
    if start and end:
        df = df.loc[start:end]
    return df
    

def parse_json_to_dataframe(symbol: str, path: str) -> pd.DataFrame:
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

def parse_csv_to_dataframe(symbol: str, path) -> pd.DataFrame:
    file_path = f"{path}/{symbol}.csv"
    data = pd.read_csv(file_path)
    
    # Parse time column as DateTimeIndex
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)   
    data.name = symbol
    
    return data