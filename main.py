import pandas as pd
from datetime import date
from Twitter_Sentiment import Tweets
import streamlit as st
import plotly.express as px
import numpy as np

st.title('Real Time Sentiment of the Social Media')
with st.sidebar:
    text = st.text_input(
        "Enter the Company / Hashtag / Public Figure name to get real time Social Media Sentiment",
        "Dish Network",
        key="Dish Network",
    )

tweets_object = Tweets(text)


# @st.cache_data(show_spinner="Fetching data from Twitter...")
def getting_data(_obj):
    return _obj.get_tweets()


tweets = getting_data(tweets_object)

overall_sentiment = \
    tweets[(tweets['Sentiment'] != 'Not Defined') & (tweets['Sentiment'] != 'Neutral')]['Sentiment'].mode()
number_of_tweets = len(tweets)
# print(overall_sentiment)
col1 , col2 = st.columns(2)


def emoji(kpi):
    if kpi == 'Positive':
        return "\N{grinning face}"
    elif kpi == 'Negative':
        return "\N{angry face}"
    else:
        return "\N{unamused face}"


with col1:
    st.metric('Overall Sentiment', emoji(overall_sentiment.values[0]), overall_sentiment.values[0])
with col2:
    st.metric('Total Tweets', number_of_tweets)

bar_df = tweets[['Tweet', 'Polarity']].groupby('Polarity').count()
bar_df.reset_index(inplace=True)


def color(x):
    if x < -30:
        return 'red'
    elif x > 30:
        return 'green'
    else:
        return 'yellow'


bar_df['color'] = bar_df['Polarity'].apply(lambda x: color(x))
bar_df['category'] = [str(i) for i in bar_df['Polarity']]
# st.write(bar_df)
fig = px.bar(bar_df, x='Polarity', y='Tweet',
             color='category',
             color_discrete_sequence=list(bar_df['color']),
             height=800, width=800)

st.plotly_chart(fig, sharing="streamlit", theme="streamlit")


with st.sidebar:
    kpi = st.selectbox('Select the tweets to display', ['Positive', 'Negative', 'Neutral'])
    temp = tweets[tweets['Sentiment'] == kpi]
    temp['Polarity'] = np.abs(temp['Polarity'])
    temp.sort_values(by='Polarity', inplace=True)


expander = st.expander("See all " + kpi + " records" + emoji(kpi))
with expander:
    st.dataframe(temp[['Tweet']], width=800, height=1000, use_container_width=True)