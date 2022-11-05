from datetime import date, timedelta
from plotly import graph_objs as go
import pandas as pd, numpy as np, streamlit as st, yfinance as yf
from cmc_api import live_price, daily_change, weekly_change, marketcap, week_before, past_month, daily_volume, daily_volume_change
import requests
import datetime
from calendar import month_name


LOGO = "https://cryptologos.cc/logos/axie-infinity-axs-logo.png"

st.set_page_config(page_title=" AXS Price Analysis & Prediction", page_icon=LOGO, layout="wide")



#css = 'D:\mynamejeff\TC Playground\'
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
current, weekly, mcap  = st.columns(3)
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

weekly.metric(
    label="Weekly Change",
    value= f"{round(weekly_change,2)}%",
    delta= f"{round(weekly_change - daily_change*2, 2)}%"
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
forecast_btn = st.button("Forecast Future Price")
st.write("##")
st.write("##")


# f = Forecaster(y=axs["close"], current_dates=axs["time"])
# f.set_test_length(.2) 
# f.generate_future_dates(2600) #3 months ahead
# f.tf_model = import_model

#forecasted.drop("Unnamed:0", axis=1)
#forecasted.drop("Unnamed:0", axis=1, inplace=True)

forecast = pd.read_csv("forecast.csv")

def display_forecast(forecast, range):

    forecast.DATE = pd.to_datetime(forecast.DATE)
    
    st.write("##")
    st.markdown("""---""")

    st.title("AXS Forecasted Price")
    st.subheader(f"Next {range} days forecast:")


    months = forecast.DATE.dt.month.unique().tolist()

    forecast_months = []

    for month in months:
        cur_month = forecast[forecast.DATE.dt.month==month]
        forecast_months.append(cur_month)

    first_month  = forecast_months[1]
    second_month = forecast_months[2]
    third_month  = forecast_months[3]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.DATE, y=forecast.lstm_default, name="Forecast", line=dict(color="#0095e8", width=7)))
    fig.layout.update(title="AXS-USD")
    fig.update_xaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
    fig.update_yaxes(griddash='dash', gridwidth=1, gridcolor='#535566')
    
    for month in forecast_months[:3]:
        fig.add_annotation(
        x=month.DATE[month.lstm_default.idxmin()],
        y=month.lstm_default.min(),
        xref="x",
        yref="y",
        text="Buy",
        showarrow=True,
        font=dict(
            family="sans-serif, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="White",
        ax=30,
        ay=30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )
        

    fig.add_annotation(
    x=forecast.DATE[forecast.lstm_default.idxmin()],
    y=forecast.lstm_default.min(),
    xref="x",
    yref="y",
    text="Strong Buy",
    showarrow=True,
    font=dict(
        family="sans-serif, monospace",
        size=16,
        color="#ffffff"
        ),
    align="center",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="White",
    ax=30,
    ay=30,
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
    bgcolor="#ff7f0e",
    opacity=0.9
    )

    fig.add_annotation(
    x=forecast.DATE[forecast.lstm_default.idxmax()],
    y=forecast.lstm_default.max(),
    xref="x",
    yref="y",
    text="Sell",
    showarrow=True,
    font=dict(
        family="sans-serif, monospace",
        size=16,
        color="#ffffff"
        ),
    align="center",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="White",
    ax=-50,
    ay=-35,
    bordercolor="Green",
    borderwidth=2,
    borderpad=4,
    bgcolor="Green",
    opacity=0.8
    )
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

    tmpm = forecast.loc[forecast.DATE == nxtmonth]
    monthforecast = price_change(live_price, tmpm.lstm_default.iloc[0])

    tmpw = forecast.loc[forecast.DATE == nxtweek]
    weeklyforecast = price_change(live_price, tmpw.lstm_default.iloc[0])


    movement = round(((tmpm.lstm_default.iloc[0] - live_price )/abs(live_price)) * 100, 2)

    if movement > 9: 
        pricemovement = "Uptrend"
    elif movement < -9:
        pricemovement = "Downtrend"
    else: 
        pricemovement = "Neutral"


    # neutral movement, downtrend, uptrend

    analysis.markdown(f"""
            # Forecast Analysis:\n
            - ### Lowest price is \${round(forecast.lstm_default.min(),2)} on {forecast.DATE[low_date]}
            - ### Highest price is \${round(forecast.lstm_default.max(),2)} on {forecast.DATE[high_date]}

            - ### Price Forecast for the next 7days:  {weeklyforecast}
            - ### Price Forecast at the end of the month:  {monthforecast}

            ## Forecast Analysis implies that AXS will be in a **{pricemovement} price movement in the upcoming month**
    """)

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
    forecast.columns = ["DATE", "FORECAST"]
    st.dataframe(forecast,use_container_width= True)
    
    csv = forecast.to_csv(index=False)
    st.write("##")
    st.download_button("Download Forecast as CSV", data=csv, file_name="timerange.csv", mime='csv')


disclaimer = "DISCLAIMER: Nothing in this site constitutes professional and or financial advice"

if forecast_btn:
    st.info(disclaimer)
    display_forecast(forecast,range=90) # <-- radio button value

st.write("##")
st.markdown("""---""")



# === News Feed  =======================================================================================
st.title("The Latest News about AXS 📰")
st.write("##")


from newscraper import get_news
try:
    news = get_news(15)
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
