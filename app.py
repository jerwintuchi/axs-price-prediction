from datetime import date, timedelta
from plotly import graph_objs as go
import matplotlib.dates as mpl_dates
import pandas as pd, numpy as np, streamlit as st, yfinance as yf, pandas_ta as ta
#from cmc_api import live_price, daily_change, weekly_change, marketcap, week_before, past_month, daily_volume, daily_volume_change
import requests
import datetime
from calendar import month_name
import copy

LOGO = "https://cryptologos.cc/logos/axie-infinity-axs-logo.png"
st.set_page_config(page_title=" AXS Price Analysis & Prediction", page_icon=LOGO, layout="wide")


#=================== CMC API Requests ===============================================================
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
parameters = {
  'symbol':'AXS',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': 'ff8378c1-34a5-41b0-8313-d0b579dc59de'
}

session = Session()
session.headers.update(headers)
try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)

quote = data["data"]["AXS"][0]["quote"]["USD"]
live_price = quote["price"]
daily_change = quote["percent_change_24h"]
weekly_change = quote["percent_change_7d"]
past_month = quote["percent_change_30d"]
week_before = past_month/4
#market_cap = int(quote["market_cap"])
daily_volume ='{:,}'.format(int(quote["volume_24h"])) 
marketcap = '{:,}'.format(int(quote["market_cap"]))
daily_volume_change = quote["volume_change_24h"]
#===================================================================================================



#css = r'D:\mynamejeff\forecaxst\.latest\styles.css'
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#Background image 
#page_bg_img = """
#<style>
#[data-testid="stAppViewContainer"]{
#background-image: url("https://images.unsplash.com/photo-1614850523011-8f49ffc73908?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80");
#background-size: cover;
#}
#</style>
#"""
#st.markdown(page_bg_img, unsafe_allow_html=True)
# # Background images
#https://images.hdqwalls.com/wallpapers/blue-white-material-design-4k-up.jpg
#https://images.unsplash.com/photo-1614850523011-8f49ffc73908?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80


# Date time constants 
TODAY = date.today().strftime("%Y-%m-%d")
WEEK_AGO = date.today() - timedelta(days=7)
WEEK_AGO = WEEK_AGO.strftime("%Y-%m-%d")

# Live Display datasets
# 2020 to current date

display_data = yf.download("AXS-USD", start="2021-02-01", end=TODAY, interval="1d")
display_data.reset_index(inplace=True)
display_data_w = yf.download("AXS-USD", start=WEEK_AGO, end=TODAY, interval="1d")
display_data_w.reset_index(inplace=True)


# BTC = yf.download("BTC-USD", start="2014-01-01", end=TODAY)
# BTC.reset_index(inplace=True)
# print(BTC.info())
# df_train = BTC[["Date", "Close"]]
# df_train = df_train.rename(columns ={"Date":"ds", "Close":"y"})


# ====== AXS Price Prediction and Analysis ===========================================================================================

st.title("AXS Price Prediction and Analysis")
# animations from lottie library
def load_lt(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
#lt_down = load_lt("https://assets8.lottiefiles.com/private_files/lf30_mmqrzxld.json")

# Triple metrics 

current, mcap, weekly  = st.columns(3)
current.metric(
    label="Current Price",
    value =f"${round(live_price,2)}",
    delta =f"{round(daily_change,2)}%",
)

mcap.metric(
    label="Trading Volume",
    value= "$"+daily_volume,
    delta = f"{round(daily_volume_change,2)}%"
)

if weekly_change > 0:
    weekly.metric(
        label="Weekly Change",
        value= f"+{round(weekly_change,2)}%"
    )
else:   
    weekly.metric(
        label="Weekly Change",
        value= f"{round(weekly_change,2)}%"
    )

# Formating string for the summary and shit
if daily_change < 0:
    str_daily = f"down {round(daily_change,2)}"  
else: str_daily = f"up {round(daily_change,2)}"

if weekly_change < 0:
    str_weekly = f"down {round(weekly_change,2)}"  
else: str_weekly = f"up {round(weekly_change,2)}"

if past_month < 0:
    str_monthly = f"down {round(past_month,2)}"  
else: str_monthly = f"up {round(past_month,2)}"

week_min = round(display_data_w.Low.min(),2)
week_max = round(display_data_w.High.max(),2)

st.write("##")
# Summary

st.header("Overview")
summary,_ = st.columns([7,1])
#str = f'Overview:\nThe Axie Infinity Shard (AXS) price today is **\${round(live_price,2)}**,  with a 24-hour trading volume of **\${daily_volume}**.  AXS is **{str_daily}%** in the last 24 hours,  **{str_weekly}%** in the last week,  **{str_monthly}%** in the past month with a live market cap of **\${marketcap}**.'

overview = f"""
        <p id = overview> The Axie Infinity Shard (AXS) price today is <b>${round(live_price,2)}</b>,  with a 24-hour trading volume of <b>${daily_volume}</b>.  AXS is <b>{str_daily}%</b> in the last 24 hours,  <b>{str_weekly}%</b> in the last week,  <b>{str_monthly}%</b> in the past month with a live market cap of <b>${marketcap}</b>.</p>
"""

summary.markdown(overview, True)

#summary.header(str)
st.write("##")
st.markdown("""---""")
st.write("##")

# === Current and historical Price Chart 📈 =======================================================================================
st.title("Current and historical Price Chart 📈")

#long_plot, week_plot = st.columns(2)


fig = go.Figure()

fig.add_trace(go.Candlestick(x=display_data.Date, open=display_data.Open, high=display_data.High, close=display_data.Close, low=display_data.Low))
fig.layout.update(title="AXS-USD (1d Intervals)")
fig.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig.update_layout(height=800)

st.title("All Time Chart")
st.plotly_chart(fig, True)

display_data["ksi"] = ta.rsi(display_data.Close,length=6)
display_data["ma"] = display_data.Close.rolling(window=13).mean()
display_data["ma5"] = display_data.Close.rolling(window=5).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=display_data.Date, y=display_data.Close, name="Forecast", line=dict(color="#0095e8", width=7)))
fig.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')

fig.add_trace(go.Scatter(x=display_data.Date, 
                        y=display_data['ma'], 
                        opacity=0.7, 
                        line=dict(color='orange', width=3), 
                        name='13d MA'))

fig.add_trace(go.Scatter(x=display_data.Date, 
                        y=display_data['ma5'], 
                        opacity=0.7, 
                        line=dict(color='red', width=2), 
                        name='5d MA'))
fig2 = go.Figure() #RSI Chart
fig2.add_trace(go.Scatter(x=display_data.Date, y=display_data.ksi, name="RSI", line=dict(color="#0095e8", width=3)))
fig2.layout.update(title="RSI",title_font_size=35,title_x=0.5)
fig2.update_xaxes(griddash='dash', gridwidth=0, gridcolor='#535566')
fig2.update_yaxes(griddash='dash', gridwidth=0, gridcolor='#535566')
st.plotly_chart(fig2, True)

#====================================================SUPPORT AND RESISTANCE==========================================================

def supportlvl(display_data,i):
    #if the previous 2 candles(1st and 2nd candle) is less than the 3rd candle (df['Low'][i]) and the succeeding 2 candles is greater than 3rd candle (df['Low'][i])
    #then it is the supportlvl
  support = display_data['Low'][i] < display_data['Low'][i-1] and display_data['Low'][i] < display_data['Low'][i+1] and display_data['Low'][i+1] < display_data['Low'][i+2] and display_data['Low'][i-1] < display_data['Low'][i-2]
  return support

def resistancelvl(display_data,i):
    #same logic with supportlvl except that it is reversed
  resistance = display_data['High'][i] > display_data['High'][i-1]  and display_data['High'][i] > display_data['High'][i+1] and display_data['High'][i+1] > display_data['High'][i+2] and display_data['High'][i-1] > display_data['High'][i-2]
  return resistance

# Initialize empty lists for support and resistance levels
support = []
resistance = []

def isFar(value, levels, display_data):
    ave = np.mean(display_data['High'] - display_data['Low'])
    return np.sum([abs(value-level)<ave for _,level in levels])==0

levels = []

for i in range(2, display_data.shape[0] - 2):
    if supportlvl(display_data, i):
        low = display_data['Low'][i]
        if isFar(low, levels, display_data):
            support.append(low)
    elif resistancelvl(display_data, i):
        high = display_data['High'][i]
        if isFar(high, levels, display_data):
            resistance.append(high)

# Create a Plotly figure
fig3 = go.Figure()
# Add a candlestick chart of the data
fig3.add_trace(go.Candlestick(x=display_data['Date'], open=display_data['Open'], high=display_data['High'], low=display_data['Low'], close=display_data['Close']))



# Create a threshold variable to set the minimum distance between lines
threshold = 0.05

# Add support levels to the figure
for i in range(len(support)):
    index = display_data.loc[display_data['Low']==support[i]].index[0]
    try:
        next_support = support[i+1]
        next_support_index = display_data.loc[display_data['Low']==next_support].index[0]
        x1 = display_data['Date'][next_support_index]
    except:
        x1 = display_data['Date'].iloc[-1]
    fig3.add_shape(
        type='line',
        x0=display_data['Date'][index],
        y0=support[i],
        x1=x1,
        y1=support[i],
        line=dict(color='green', width=3, dash='dot')
    )
# Add resistance levels to the figure
for i in range(len(resistance)):
    indexr = display_data.loc[display_data['High']==resistance[i]].index[0]
    try:
        next_resistance = resistance[i+1]
        next_resistance_index = display_data.loc[display_data['High']==next_resistance].index[0]
        x1 = display_data['Date'][next_resistance_index]
    except:
        x1 = display_data['Date'].iloc[-1]
    fig3.add_shape(
        type='line',
        x0=display_data['Date'][indexr],
        y0=resistance[i],
        x1=x1,
        y1=resistance[i],
        line=dict(color='red', width=3, dash='dot')
    )

# Removing duplicates values
support = list(set(support))
resistance = list(set(resistance))



fig3.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig3.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig3.layout.update(title="Support and Resistance Levels",title_font_size=35,title_x=0.5)
fig3.update_layout(height=1000)

fig3.update_layout(
    dragmode="drawopenpath",
    newshape_line_color="cyan",
    title_text="You can draw within this chart.",
)
config = dict(
    {
        "scrollZoom": True,
        "displayModeBar": True,
        # 'editable'              : True,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
        "toImageButtonOptions": {"format": "svg"},
    }
)
st.plotly_chart(fig3, True)

# fig2.add_trace(go.Scatter(x=display_data_w.Date, y=display_data_w.Close, name="Price"))
# fig2.layout.update(title="AXS-USD (1d Intervals)")
# week_plot.subheader("Last 7 days")
# week_plot.plotly_chart(fig2)

fig2 = go.Figure()
st.title("Last 7 days Chart")
st.markdown(f"""<p id='weekly_summary'> For this week, the highest price of AXS is  <strong>${week_max}</strong>, and has been as low as  <strong>${week_min}</strong>. </p>""", unsafe_allow_html=True) 

fig2.add_trace(go.Candlestick(x=display_data_w.Date, open=display_data_w.Open, high=display_data_w.High, close=display_data_w.Close, low=display_data_w.Low))
fig2.layout.update(title="AXS-USD (1d Intervals)", xaxis_rangeslider_visible = False)
fig2.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig2.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
fig2.update_layout(height=650)
st.plotly_chart(fig2, True)

st.write("##")

disclaimer = "DISCLAIMER: Nothing in this site constitutes professional and or financial advice. All buying & selling signals are approximate measures from the predictive model within the 3 month period."
st.info(disclaimer)
agree = st.checkbox("I agree with the disclaimer")
st.write("##")

placeholder = st.empty()
placeholder.button("Forecast Future Price", key="ph", disabled=True, help="Please agree with the disclaimer first")

#csv = r'D:\mynamejeff\forecaxst\.latest\forecast.csv'
forecast = pd.read_csv("forecast.csv")

def display_forecast(forecast):

    forecast.DATE = pd.to_datetime(forecast.DATE)
    
    months = forecast.DATE.dt.month.unique().tolist()
    forecast_months = []
    for month in months:
        cur_month = forecast[forecast.DATE.dt.month==month]
        forecast_months.append(cur_month)
        
    month0_str = f"{month_name[forecast_months[0].DATE.iloc[0].month]} {forecast_months[0].DATE.iloc[0].year}"
    monthend_str = f"{month_name[forecast_months[-1].DATE.iloc[0].month]} {forecast_months[-1].DATE.iloc[0].year}"
    range_str = f"Forecast from {month0_str} to {monthend_str}"

    first_month  = forecast_months[1]
    second_month = forecast_months[2]
    third_month  = forecast_months[3]
    
    st.write("##")
    st.markdown("""---""")
    st.title("AXS Forecasted Price")
    st.subheader(range_str)

    forecast["ma"] = forecast.lstm_default.rolling(window=300).mean()
    forecast["ma4"] = forecast.lstm_default.rolling(window=100).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.DATE, y=forecast.lstm_default, name="Forecast", line=dict(color="#0095e8", width=7)))
    fig.layout.update(title="AXS-USD")
    fig.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
    fig.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
    
    fig.add_trace(go.Scatter(x=forecast.DATE, 
                         y=forecast['ma'], 
                         opacity=0.7, 
                         line=dict(color='orange', width=3), 
                         name='8d MA'))
    fig.add_trace(go.Scatter(x=forecast.DATE, 
                         y=forecast['ma4'], 
                         opacity=0.7, 
                         line=dict(color='red', width=2), 
                         name='4d MA'))

    
# BUY/SELL SIGNAL (OMMITED)    
#     for month in forecast_months[:3]:
#         fig.add_annotation(
#         x=month.DATE[month.lstm_default.idxmin()],
#         y=month.lstm_default.min(),
#         xref="x",
#         yref="y",
#         text="Buy",
#         showarrow=True,
#         font=dict(
#             family="sans-serif, monospace",
#             size=16,
#             color="#ffffff"
#             ),
#         align="center",
#         arrowhead=2,
#         arrowsize=1,
#         arrowwidth=2,
#         arrowcolor="White",
#         ax=30,
#         ay=30,
#         bordercolor="#c7c7c7",
#         borderwidth=2,
#         borderpad=4,
#         bgcolor="#ff7f0e",
#         opacity=0.8
#         )
        

# #     fig.add_annotation(
# #     x=forecast.DATE[forecast.lstm_default.idxmin()],
# #     y=forecast.lstm_default.min(),
# #     xref="x",
# #     yref="y",
# #     text="Strong Buy",
# #     showarrow=True,
# #     font=dict(
# #         family="sans-serif, monospace",
# #         size=16,
# #         color="#ffffff"
# #         ),
# #     align="center",
# #     arrowhead=2,
# #     arrowsize=1,
# #     arrowwidth=2,
# #     arrowcolor="White",
# #     ax=30,
# #     ay=30,
# #     bordercolor="#c7c7c7",
# #     borderwidth=2,
# #     borderpad=4,
# #     bgcolor="#ff7f0e",
# #     opacity=0.9
# #     )

#     fig.add_annotation(
#     x=forecast.DATE[forecast.lstm_default.idxmax()],
#     y=forecast.lstm_default.max(),
#     xref="x",
#     yref="y",
#     text="Sell",
#     showarrow=True,
#     font=dict(
#         family="sans-serif, monospace",
#         size=16,
#         color="#ffffff"
#         ),
#     align="center",
#     arrowhead=2,
#     arrowsize=1,
#     arrowwidth=2,
#     arrowcolor="White",
#     ax=-50,
#     ay=-35,
#     bordercolor="Green",
#     borderwidth=2,
#     borderpad=4,
#     bgcolor="Green",
#     opacity=0.8
#     )
    
    fig.update_layout(height=500, width=1000)
    plot, analysis = st.columns([4,3])
    plot.plotly_chart(fig)
 

    def price_change(start, end):
        price_change = round(((end - start )/abs(start)) * 100, 2)
        if price_change > 0:
            price_change = f"+{price_change}" 
        return f"{price_change}%"

    high_date = forecast.lstm_default.idxmax()
    low_date = forecast.lstm_default.idxmin()

    neckweek = datetime.datetime.now() + datetime.timedelta(days=8)
    nxtweek = datetime.datetime.strptime(neckweek.strftime("%m-%d-%Y"), "%m-%d-%Y")

    neckmonth = datetime.datetime.now() + datetime.timedelta(days=31)
    nxtmonth = datetime.datetime.strptime(neckmonth.strftime("%m-%d-%Y"), "%m-%d-%Y")
    print(nxtmonth)

    tmpm = forecast.loc[forecast.DATE == nxtmonth]
    # monthforecast = price_change(live_price, tmpm.lstm_default.iloc[0])
    print(tmpm)

    tmpw = forecast.loc[forecast.DATE == nxtweek]
    # weeklyforecast = price_change(live_price, tmpw.lstm_default.iloc[0])
    
    
    monthnow = datetime.datetime.now().month
    monthend = forecast[forecast.DATE.dt.month==monthnow].lstm_default.iloc[-1]
    # monthend_change = price_change(live_price, monthend)

    # movement = round(((tmpm.lstm_default.iloc[0] - live_price )/abs(live_price)) * 100, 2)

    # if movement > 9: 
    #     pricemovement = "Uptrend"
    # elif movement < -9:
    #     pricemovement = "Downtrend"
    # else: 
    #     pricemovement = "Neutral"


    # neutral movement, downtrend, uptrend

    # analysis.markdown(f"""
    #         # Forecast Analysis:\n
    #         - ### Lowest price is \${round(forecast.lstm_default.min(),2)} on {forecast.DATE[low_date]}
    #         - ### Highest price is \${round(forecast.lstm_default.max(),2)} on {forecast.DATE[high_date]}

            
    # """)

     ### Price Forecast for the next 7 days:  {weeklyforecast}
     ### Price Forecast at the end of the month:  {monthend_change}
    ## Forecast Analysis implies that AXS will be in a **{pricemovement} price movement in the upcoming month**
    st.write("##")
    st.markdown("""---""")

    st.markdown("# Price Forecast for the next 3 months")
    month0_col, month1_col, month2_col = st.columns(3)
    #_, month3_col ,_ = st.columns(3)

    month0_col.markdown(f"""
            ## Month of {month_name[months[1]]}
            - ### Starting price: ${round(first_month.lstm_default.iloc[0], 2)}
            - ### End price: ${round(first_month.lstm_default.iloc[-1], 2)}
            - ### Change: {price_change(first_month.lstm_default.iloc[0], first_month.lstm_default.iloc[-1])}
            - ### Average Price: ${round(first_month.lstm_default.mean(),2)}
    """)

    month1_col.markdown(f"""
            ## Month of {month_name[months[2]]}
            - ### Starting price: ${round(second_month.lstm_default.iloc[0], 2)}
            - ### End price: ${round(second_month.lstm_default.iloc[-1], 2)}
            - ### Change: {price_change(second_month.lstm_default.iloc[0], second_month.lstm_default.iloc[-1])}
            - ### Average Price: ${round(second_month.lstm_default.mean(),2)}
    """)
    
    month2_col.markdown(f"""
            ## Month of {month_name[months[3]]}
            - ### Starting price: ${round(third_month.lstm_default.iloc[0], 2)}
            - ### End price: ${round(third_month.lstm_default.iloc[-1], 2)}
            - ### Change: {price_change(third_month.lstm_default.iloc[0], third_month.lstm_default.iloc[-1])}
            - ### Average Price: ${round(third_month.lstm_default.mean(),2)}
    """)

    #month3_col.markdown(f"""
    #         ## Month of {month_name[months[3]]}
    #         - ### Starting price: {round(third_month.lstm_default.iloc[0], 2)}
    #         - ### End price: ${round(third_month.lstm_default.iloc[-1], 2)}
    #         - ### Change: {price_change(third_month.lstm_default.iloc[0], third_month.lstm_default.iloc[-1])}
    #         - ### Average Price: ${round(third_month.lstm_default.mean(),2)}
    # """)

    

    st.write("##")
    st.markdown("""---""")

    st.markdown("# Individual Forecast Table")
    #table.dataframe(forecast)

    # go.Table(
    # header=dict(values=list(df.columns),
    #             fill_color='paleturquoise',
    #             align='left'),
    # cells=dict(values=df.transpose().values.tolist(),
    #            fill_color='lavender',
    #            align='left'))

    fig2 = go.Figure()

    # fig2.add_trace(go.Table(header=dict(values=list(forecast.columns),
    #                                     align='left',
    #                                     fill_color='paleturquoise'),

    #                         cells=dict(values=forecast.transpose().values.tolist(),
    #                                     align='left',
    #                                     fill_color='lavender')
    #                         ))
    
    forecast.drop("Unnamed: 0",axis=1, inplace=True)
    try:
        forecast.columns = ["DATE", "FORECAST", "MA8", "MA4"]
        st.dataframe(forecast,use_container_width= True)

        filename = f"{forecast_months[0].DATE.iloc[0].month}-{forecast_months[0].DATE.iloc[0].year}_to_{forecast_months[-1].DATE.iloc[0].month}-{forecast_months[-1].DATE.iloc[0].year}_Forecast.csv"
        csv = forecast.to_csv(index=False)
        st.write("##")
        st.download_button("Download Forecast as CSV", data=csv, file_name=filename, mime='csv')
    except(Exception):
        st.write("refresh browser")

if agree:
    placeholder.empty()
    foreast_btn = st.button("Forecast Future Price")

    # try:
    if foreast_btn:
        display_forecast(forecast) # <-- radio button value
    # except(Exception):
    #         st.markdown('e')

st.write("##")
st.write("##")


# f = Forecaster(y=axs["close"], current_dates=axs["time"])
# f.set_test_length(.2) 
# f.generate_future_dates(2600) #3 months ahead
# f.tf_model = import_model

#forecasted.drop("Unnamed:0", axis=1)
#forecasted.drop("Unnamed:0", axis=1, inplace=True)

        


# try:
#     if forecast_btn:
#         display_forecast(forecast) # <-- radio button value
# except(Exception):
#     pass
    

st.write("##")
st.markdown("""---""")



# === News Feed  =======================================================================================
st.title("The Latest News about AXS 📰")
st.write("##")

# def sortbydate(news_list):
#     for i in news_list:
#         i["date"] = datetime.datetime.strptime(i["date"], '%B %d, %Y')
#     news_list.sort(key = lambda x:x['date'], reverse=True)

#     for i in news_list:
#         i["date"] = datetime.datetime.strftime(i["date"], "%B %d, %Y")

#     return news_list


from newscraper import get_news
newsc = get_news(11) 
news = copy.copy(newsc)
try:
    for i in news:
        i["date"] = datetime.datetime.strptime(i["date"], '%B %d, %Y')
    news.sort(key = lambda x:x["date"], reverse=True)
    for i in news:
        i["date"] = datetime.datetime.strftime(i["date"], "%B %d, %Y")  
except(Exception):
    st.error("Failed to fetch news, Please restart your browser.")


backup_thumbnail = "https://cryptoslate.com/wp-content/uploads/2021/08/axs-750x.jpg"

def display_news(num_of_news=15):
    i = 0
    for new in news:
        with st.container():
            url = news[i]["link"]
            thumbnail, text = st.columns([1,3])
            with thumbnail:
                try:
                    st.image(news[i]["thumbnail"], width=350, use_column_width="auto")
                except(Exception): # Not all articles have thumbnails
                    st.image(backup_thumbnail, width=350, use_column_width="auto")

            with text:
                st.subheader(news[i]['title'])
                st.subheader(f'By {news[i]["publisher"]}, Published {news[i]["date"]}\n[Read more about the article ➜](%s)' % url)

        st.markdown(""" --- """)
        i+=1
        if i >= num_of_news: break

display_news()

# st.subheader("Predictive Model's Dataset")
# tail, descr = st.columns(2)
# tail.dataframe(axs.tail())
# descr.dataframe(axs.describe())

# #prophet
# def plot_prediction():

#     model = Prophet()
#     model.fit(df_train)
#     future = model.make_future_dataframe(periods=93 *24, freq="H")
#     future_preds = model.predict(future)    

#     st.subheader("Forecasted Chart")
#     futureplot = plot_plotly(model,future_preds)
#     st.plotly_chart(futureplot)

#     st.subheader("Forecast components")
#     fig2 = model.plot_components(future_preds)
#     st.write(fig2)

footer = """
        <p id = footlong> Made by <span id = footshort> TonyongBayawak  </span> with love ❤️</p>
        <p id = footlong>Copyright © 2022 All Rights Reserved by group TonyongBayawak<p>
"""

st.markdown(footer, unsafe_allow_html=True)
