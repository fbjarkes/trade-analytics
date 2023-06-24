import pandas as pd
import streamlit as st
import plotly.graph_objects as go


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
    data = pd.read_csv(file)    
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    #data = data.resample('D').ffill() TODO: resample to daily?
    # filter date from 2020-01-01
    data = data.loc['2020-01-01':]
    #data.name = symbol´
    return data

def main():
    st.write('Trade CSV analysis')



 # Upload another CSV file for the OHLC chart
uploaded_file = st.sidebar.file_uploader('Upload OHLC CSV file:')
if uploaded_file is not None:
    ohlc_df = parse_uploaded_csv(uploaded_file)
    print(f"Plotting number of rows: {len(ohlc_df)} with start date {ohlc_df.index[0]} and end date {ohlc_df.index[-1]}")
    st.write('## OHLC Chart')
    plot_ohlc_chart(ohlc_df)


if __name__ == '__main__':
    main()