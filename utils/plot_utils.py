

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
    
