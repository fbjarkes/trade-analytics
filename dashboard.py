from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import utils.stats_utils as stats_utils
import utils.data_utils as data_utils
import utils.func_utils as func_utils

NO_TRADES_DF = pd.DataFrame(columns=['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl'])

PROVIDER_CONFIG = {
    'tv': '/Users/fbjarkes/Bardata/tradingview',
    'alpaca': '/Users/fbjarkes/Bardata/alpaca-v2',
    'ib': '/Users/fbjarkes/Bardata/ibfile'
}

def main():
    st.set_page_config(layout="wide", page_title='CSV Trade Analytics')   
    st.session_state.start_eq = 10000    
    st.sidebar.markdown(f"## Bardata options")
    provider = st.sidebar.selectbox('Select data provider:', ['alpaca', 'tv', 'ib'])
    st.session_state.timeframe = st.sidebar.selectbox('Select timeframe', ['15min', '30min', 'day'])    
    # path = st.sidebar.text_input('Data path', value=PROVIDER_CONFIG[provider])
    # if st.sidebar.button('Load data'):
    #     tickers_dict = load_tickers_data(trades, f"{PROVIDER_CONFIG[provider]}/{tf}", provider=provider)
    # ==== CSV files upload ====
    st.sidebar.markdown(f"## Upload trades files")
    trades_csv_data = st.sidebar.file_uploader(f"Trades CSV", type=['csv'])
    if trades_csv_data:
        trades = data_utils.load_trades(trades_csv_data)
        tickers_dict = data_utils.load_tickers_data(trades, f"{PROVIDER_CONFIG[provider]}/{st.session_state.timeframe}", provider=provider)
        metric_trades = data_utils.apply_metrics(trades, tickers_dict)
        #trades, tickers_dict = init_data(trades_csv_data, f"{PROVIDER_CONFIG[provider]}/{tf}", provider=provider)
        st.session_state.trades = metric_trades
        st.sidebar.text(f"File '{trades_csv_data.name}'")
    else:
        st.session_state.trades = NO_TRADES_DF
        st.session_state.filtered_trades = NO_TRADES_DF
        st.sidebar.text('No file selected')
            
    
    # ==== Filter inputs ====
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
    with col2:
        if st.session_state.trades.empty:
            start_date = datetime(datetime.now().year, 1, 1).date()
            end_date = datetime.now().date()
        else:
            start_date = st.session_state.trades.index[0].date()
            end_date =  st.session_state.trades.index[-1].date()
        st.session_state.start_date = st.date_input('Filter Start Date', value=start_date)        
        st.session_state.end_date = st.date_input('Filter End Date', value=end_date)  # NOTE: just use last start date
        st.session_state.selected_metric = st.selectbox('Select Rank Metric:', stats_utils.SELECTABLE_METRICS)
        st.session_state.selected_rank = st.selectbox('Select Top Ranked:', [1,2,3,4,5,6,7,8,9,10])
        st.session_state.symbols = [sym.upper() for sym in st.text_input('Symbols (comma separated):').split(',')]
        st.write(f"Filter by Index (normalized)")
        # TODO: select predefined indexes here? (and load and prep. when selected)
        st.session_state.selected_index_metric_normalized = st.selectbox('Select Index Metric:', stats_utils.INDEX_METRICS_NORMALIZED)
        st.write(f"Filter by Index")
        with st.expander('Index Metric'):
            # TODO: select predefined indexes here? (and load and prep. when selected)
            st.session_state.selected_index_metric = st.selectbox('Select Metric:', stats_utils.INDEX_METRICS)
            st.session_state.selected_index_value = st.number_input('Index Value:', value=0.0)        
        st.divider()
        st.write(f"Equity Curve Filter")
        st.session_state.selected_ec_filter = st.selectbox('Select filter:', stats_utils.EC_FILTER_METRICS)
        
        st.divider()
        st.write(f"View trades by row numbers ({len(st.session_state.filtered_trades)} rows):")
        st.session_state.table_row_first = st.text_input('First row', value=len(st.session_state.filtered_trades)-500)
        st.session_state.table_row_last = st.text_input('Last row', value=len(st.session_state.filtered_trades))
        
    filter_trades = func_utils.compose(
        partial(stats_utils.filter_rank, metric=st.session_state.selected_metric, rank=st.session_state.selected_rank), 
        partial(stats_utils.filter_symbols, symbols=st.session_state.symbols), 
        partial(stats_utils.filter_start_date, date=st.session_state.start_date))
    
    if st.session_state.trades.empty:
        st.session_state.filtered_trades = NO_TRADES_DF
    else:    
        # TODO: run with click of button?
        st.session_state.filtered_trades = filter_trades(st.session_state.trades)
        if st.session_state.selected_ec_filter != 'NONE':
            # TODO: add to compose?
            st.session_state.filtered_trades = stats_utils.filter_equity_curve(st.session_state.filtered_trades, st.session_state.selected_ec_filter)
                
    # ==== TABLE ====
    d1, d2, d3 = st.columns([0.1, 0.8, 0.1])
    with d2:
        if st.session_state.filtered_trades.empty:        
            #st.dataframe(NO_TRADES_DF, use_container_width=True)        
            st.write('No trades found')
        else:
            #TODO: need to set a max size?
            st.dataframe(st.session_state.filtered_trades.iloc[int(st.session_state.table_row_first):int(st.session_state.table_row_last)-1], use_container_width=True)
                
    # ==== Baseline ====
    res1, res2, res3, res4 = st.columns([0.2, 0.3, 0.3, 0.2])    
    trade_stats, cum_sum = stats_utils.get_trade_stats(st.session_state.trades, st.session_state.start_eq)
    with res2:        
        st.header(f"Baseline ({len(st.session_state.trades)})")
        st.subheader('Average PnL stats:')
        for key, value in trade_stats.items():
            st.markdown(f'**{key}:** {value:.2f}')        
    with res3:
        st.subheader('Cumulative Equity curve')
        with st.spinner('Loading chart'):
            st.line_chart(cum_sum)
    
    # ==== Metric result ====
    res1, res2, res3, res4 = st.columns([0.2, 0.3, 0.3, 0.2])
    if st.session_state.filtered_trades.empty:
        t, p_value = 0, 0    
    else:
        t, p_value = stats.ttest_ind(st.session_state.trades['pnl'], st.session_state.filtered_trades['pnl'])
    trade_stats, cum_sum = stats_utils.get_trade_stats(st.session_state.filtered_trades, st.session_state.start_eq)
    trade_stats['t-value'] = t
    trade_stats['p-value'] = p_value
    with res2:        
        st.header(f"Rank {st.session_state.selected_metric} ({len(st.session_state.filtered_trades)})")
        st.subheader('Average PnL stats:')
        for key, value in trade_stats.items():
            st.markdown(f'**{key}:** {value:.2f}')           
    with res3:
        st.subheader('Cumulative Equity curve')
        with st.spinner('Loading chart'):
            st.line_chart(cum_sum)
    

    # ==== Index compare ====
    # col1, col2, col3 = st.columns([0.3, 0.4, 0.3]) 
    # with col2:
    #     uploaded_file = st.sidebar.file_uploader('Upload OHLC CSV file:')
    #     if uploaded_file is not None:
    #         df, name = parse_uploaded_csv(uploaded_file)
    #         print(f"Filtering {name} with start date {st.session_state.start_date}")
    #         df = df.loc[st.session_state.start_date:st.session_state.end_date]
    #         df = df[['close']]
    #         df.rename(columns={'close': name}, inplace=True)
            # 
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