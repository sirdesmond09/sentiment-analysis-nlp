import requests
import json
from  requests_oauthlib import OAuth1



# url = 'http://127.0.0.1:5000/'

# tweet = "I really like mondays! They are the best."

# response = requests.get(url+"predict", params={"tweet":tweet})

# print(response)
# print(response.text)

def get_tweets(*args, count=10):
    url = 'https://api.twitter.com/1.1/search/tweets.json'

    tweets = []

    for search_term in args:
        params = {"q": search_term, "lang":"en", "count":count,"tweet_mode":"extended"}

        response = requests.get(url, auth=auth, params=params)
        tweets += response.json()['statuses']
    return tweets


with open("twitter_secrets.json", 'r') as f:
    secrets = json.load(f)

auth = OAuth1(secrets['api_key'],
              secrets['api_secret'],
              secrets['access_token'],
              secrets['access_token_secret'])


tweets = get_tweets("jumia reviews")
for tweet in tweets:
    r = requests.get(url='http://127.0.0.1:5000/predict', params={"tweet":tweet['full_text']})
    print("===============")
    print(tweet['full_text'])
    print(r.text, '\n')