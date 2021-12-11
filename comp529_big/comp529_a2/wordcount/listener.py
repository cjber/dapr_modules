import tweepy
from kafka import KafkaClient


class TwitterListener(tweepy.StreamListener):
    def on_status(self, status):
        client = K
