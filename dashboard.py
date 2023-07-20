from functools import partial
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import utils


def plot_ohlc_chart(df):
    #df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)    
    #mpf.plot(df, type='candle', volume=False, style='yahoo')
    fig = go.Figure(data=go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
    # Update chart layout
    #print(fig)
    fig.update(layout_xaxis_rangeslider_visible=False)
    # Display chart using Streamlit
    st.plotly_chart(fig)
    

def parse_uploaded_csv(file):
    print(f"Parsing csv {file}")
    data = pd.read_csv(file)    
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data = data.resample('D').ffill() # TODO: best way to remove the time part of DateTimeIndex?    
    name = file.name.split('.')[0].upper()
    return data, name

@st.cache_data 
def load_trades(csv_path: str):
    print(f"Reading trades csv: {csv_path}")
    trades = utils.read_trades_csv(csv_path)
    return trades

@st.cache_data 
def init_data(trades_path: str, data_path: str):
    trades = load_trades(trades_path)      
    tickers = trades['symbol'].unique()
    
    ### DEBUG
    # 100 random tickers
    #tickers = np.random.choice(tickers, 100, replace=False)
    #print(f"Randomly selected {len(tickers)} tickers:", tickers)
    # filter trades by tickers
    #trades = trades[trades['symbol'].isin(tickers)]
    ### 
    
    if data_path:
        print(f"Loading data from '{data_path}' for {len(tickers)} tickers")
        tickers_data_df, runtime =  utils.load_json_to_df_async(data_path, tickers)
        print(f"load_json_to_df_async: {runtime:.2f} seconds")
    
        print(f"Applying metrics to tickers data ({len(tickers_data_df)})")
        tickers_data_df, runtime = utils.p_map(tickers_data_df, partial(utils.apply_rank_metric, metrics=utils.RANK_METRICS))
        print()
        print(f"p_map: {runtime:.2f} seconds")
        sum = 0   
        for df in tickers_data_df:
            sum += df.memory_usage().sum()
        print(f"Total memory usage: {sum/1000/1000:.1f} MB for {len(tickers_data_df)} tickers") 
        tickers_dict = {df.name: df for df in tickers_data_df} 
    
        print(f"Applying metrics to trades data ({len(trades)})")
        trades = utils.apply_rank(utils.RANK_METRICS, trades, tickers_dict)
    else:
        tickers_dict = {}
    
    return trades, tickers_dict

def main():
    st.set_page_config(layout="wide", page_title='CSV Trade Analytics')   
    with st.spinner('Initializing...'):        
        if 'trades' not in st.session_state and 'tickers_dict' not in st.session_state:
            #trades, tickers_dict = init_data('bbrev_trades.csv', '/Users/fbjarkes/Bardata/alpaca-v2/15min_bbrev')
            trades, tickers_dict = init_data('trades/EMA10_MR_Simple_10Perc_10EMA_Exit_trades.csv', '')
            st.session_state.trades = trades
            st.session_state.tickers_dict = tickers_dict
        
        # ==== Filter inputs ====
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col2:
            st.session_state.start_date = st.date_input('Filter Start Date', value=st.session_state.trades.index[0].date())        
            st.session_state.end_date = st.date_input('Filter End Date', value=st.session_state.trades.index[-1].date())  # NOTE: just use last start date
            st.session_state.selected_metric = st.selectbox('Select Rank Metric:', utils.SELECTABLE_METRICS)
            st.session_state.selected_rank = st.selectbox('Select Top Ranked:', [1,2,3,4,5,6,7,8,9,10])
            st.session_state.symbols = [sym.upper() for sym in st.text_input('Symbols (comma separated):').split(',')]
                    
                    
        # ==== TABLE ====
        d1, d2, d3 = st.columns([0.1, 0.8, 0.1])
        with d2:
            filter_trades = utils.compose(
            partial(utils.filter_rank, metric=st.session_state.selected_metric, rank=st.session_state.selected_rank), 
            partial(utils.filter_symbols, symbols=st.session_state.symbols), 
            partial(utils.filter_start_date, date=st.session_state.start_date))
                                   
            st.session_state.filtered_trades = filter_trades(st.session_state.trades)
            st.dataframe(st.session_state.filtered_trades.tail(500), use_container_width=True)
                    
        # ==== Baseline ====
        res1, res2, res3, res4 = st.columns([0.2, 0.3, 0.3, 0.2])
        with res2:
            st.header(f"Baseline ({len(st.session_state.trades)})")
            st.subheader('Avg. PnL')
            st.metric('Mean', st.session_state.trades['pnl'].mean())
            st.metric('Std',  st.session_state.trades['pnl'].std())
            st.metric('Max open', utils.max_open(st.session_state.trades))
        with res3:
            st.subheader('Cumulative Equity curve')
            with st.spinner('Loading chart'):
                st.line_chart(st.session_state.trades['pnl'].cumsum())
        
        # ==== Metric result ====
        res1, res2, res3, res4 = st.columns([0.2, 0.3, 0.3, 0.2])
        with res2:
            st.header(f"Rank {st.session_state.selected_metric} ({len(st.session_state.filtered_trades)})")
            st.subheader('Avg. PnL')
            st.metric('Mean', st.session_state.filtered_trades['pnl'].mean())
            st.metric('Std',  st.session_state.filtered_trades['pnl'].std())
            st.metric('Max open', utils.max_open(st.session_state.filtered_trades))        
        with res3:
            st.subheader('Cumulative Equity curve')
            with st.spinner('Loading chart'):
                st.line_chart(st.session_state.filtered_trades['pnl'].cumsum())
        # t, p-value = stats.ttest_ind(<BASELINE_PNL>, <RANK_PNL>)
        # plot cumsum

        # ==== Index compare ====
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col2:
            uploaded_file = st.sidebar.file_uploader('Upload OHLC CSV file:')
            if uploaded_file is not None:
                df, name = parse_uploaded_csv(uploaded_file)
                print(f"Filtering {name} with start date {st.session_state.start_date}")
                df = df.loc[st.session_state.start_date:st.session_state.end_date]
                df = df[['close']]
                df.rename(columns={'close': name}, inplace=True)
                
                #TODO: resample pnl series to daily and create 10 bucket sizes of number of trades
                # print IWM df with nbr_trades added as new column
                
                # TODO: resample to daily and plot with different axis
                #cum_sum = st.session_state.filtered_trades['pnl'].cumsum().resample('D').sum()                        
                #df[f"Rank {st.session_state.selected_metric} EQ"] = cum_sum                
                #st.line_chart(df)

        #ind1, ind2, ind3 = st.columns([0.2, 0.6, 0.2])

 # Upload another CSV file for the OHLC chart
# uploaded_file = st.sidebar.file_uploader('Upload OHLC CSV file:')
# if uploaded_file is not None:
#     ohlc_df = parse_uploaded_csv(uploaded_file)
#     print(f"Plotting number of rows: {len(ohlc_df)} with start date {ohlc_df.index[0]} and end date {ohlc_df.index[-1]}")
#     st.write('## OHLC Chart')
#     plot_ohlc_chart(ohlc_df)


if __name__ == '__main__':
    main()