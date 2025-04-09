from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
parameters = {
  'symbol':'AXS',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': os.getenv('X-CMC_PRO_API_KEY')
}

session = Session()
session.headers.update(headers)
try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    print(data)
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

# print(daily_volume)
# print(marketcap)