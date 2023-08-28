from requests_oauthlib import OAuth1Session
import os

from dotenv import load_dotenv

def tweeter():
    # Load environment variables
    load_dotenv()
    consumer_key = os.environ.get("twitter_api_key")
    consumer_secret = os.environ.get("twitter_api_secret_key")
    # Assuming you've added your access token and secret to the .env file
    access_token = os.environ.get("twitter_access_token")
    access_token_secret = os.environ.get("twitter_access_token_secret")
    bearer_token = os.environ.get("bearer_token")
    import tweepy

    client = tweepy.Client(bearer_token=bearer_token,
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret)
    
    return client

