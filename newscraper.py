from googlenews import GoogleNews
from newspaper import Article, ArticleException
from datetime import datetime
from time import mktime
import streamlit as st


NOW=datetime.now()
gn = GoogleNews()

#"thumbnail": "https://cryptoslate.com/wp-content/uploads/2021/08/axs-750x.jpg"
# need to try catch failed fetches and replace it with a backup thumbnail
def scrape_thumbnail(list):  
    for i in list:
        try:
            scraper = Article(url=i["link"])
            scraper.download()
            scraper.parse()
            i["thumbnail"] = scraper.top_image
        except(ArticleException): 
            list.remove(i)
    return list


@st.cache_resource()
def get_news(news_count):
    q = gn.search("Axie Infinity", when="14d")
    entries = q["entries"]
    #print(entries)
    news = []
    n=0
    for i in entries:
        title = i.title
        time = i.published_parsed
        dt=datetime.fromtimestamp(mktime(time))
        #print(dt)
        #date=f"{NOW-dt}".split(",",1)[0] + " ago" 
        date = dt.strftime("%B %d, %Y")
        article = {
            "title":title.rsplit('-', 1)[0],
            "link":i.link,
            "publisher":i.source.title,
            "date":date,#date
            "thumbnail":"",
        }
        news.append(article)
        n+=1
        if n > news_count: 
            break

    return scrape_thumbnail(news)
#news = get_news()
# print(len(news))