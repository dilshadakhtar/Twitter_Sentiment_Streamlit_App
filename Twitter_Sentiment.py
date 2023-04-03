from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Data_Cleaning import clean_message
from textblob import TextBlob
from datetime import date
import pandas as pd
import tweepy
import re
# Part-1: Authorization and Search tweets
# Getting authorization

consumer_key = ''
consumer_key_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


class Tweets:
    def __init__(self, words):
        self.words = words

    @staticmethod
    def trending():
        trends_result = api.get_place_trends(23424977)
        trends = trends_result[0]['trends']
        hashtags = []
        topic = []
        for trend in trends:
            name = trend['name']
            topic.append(name)
            if name.startswith('#'):
                hashtags.append(name)

        hashtags = pd.Series(hashtags).apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
        topics = pd.Series(topic).apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
        return hashtags, topics

    # calculate the negative, positive, neutral and compound scores, plus verbal evaluation
    def sentiment_vader(self, sentence):
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        sentiment_dict = sid_obj.polarity_scores(sentence)
        negative = sentiment_dict['neg']
        neutral = sentiment_dict['neu']
        positive = sentiment_dict['pos']
        compound = sentiment_dict['compound']

        if sentiment_dict['compound'] >= 0.05:
            overall_sentiment = "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            overall_sentiment = "Negative"

        else:
            overall_sentiment = "Neutral"

        return negative, neutral, positive, compound, overall_sentiment

    def sentiment_texblob(self, row):
        classifier = TextBlob(row)
        polarity = classifier.sentiment.polarity
        subjectivity = classifier.sentiment.subjectivity

        return polarity, subjectivity

    def sentiment(self, tweet):
        negative, neutral, positive, compound, overall_sentiment = self.sentiment_vader(tweet)
        polarity, subjectivity = self.sentiment_texblob(tweet)
        if overall_sentiment == 'Positive' and polarity > 0:
            return polarity, overall_sentiment
        elif overall_sentiment == 'Negative' and polarity < 0:
            return polarity, overall_sentiment
        elif overall_sentiment == 'Neutral' and 0.1 > polarity > -0.1:
            return 0, overall_sentiment
        else:
            return -1, 'Not Defined'

    def get_tweets(self):
        tweets = tweepy.Cursor(api.search_tweets,
                               self.words, lang="en",
                               since_id=date.today(),
                               tweet_mode='extended').items()
        list_tweets = [tweet for tweet in tweets]
        # Counter to maintain Tweet Count
        i = 1

        # we will iterate over each tweet in the
        # list for extracting information about each tweet
        df = pd.DataFrame()
        for tweet in list_tweets:
            hashtags = tweet.entities['hashtags']

            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text
            hashtext = list()
            for j in range(0, len(hashtags)):
                hashtext.append(hashtags[j]['text'])

            i = i + 1
            clean_text = clean_message(text)
            polarity, sentiment = self.sentiment(clean_text)
            polarity = polarity * 100

            try:
                if sentiment != 'Not Defined':
                    df = df.append({'Tweet': text, 'Clean Text': clean_text, 'Polarity': round(polarity, -1),
                                'Sentiment': sentiment},
                               ignore_index=True)
            except:
                pass
        df = df.drop_duplicates(subset=['Clean Text'])
        return df


if __name__ == "__main__":
    tweet_obj = Tweets('Dish Network')
    tweet_obj.get_tweets().to_csv('Tweets.csv')
