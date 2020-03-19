import json
import tweepy
import sys
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from processutils import prepare_replies

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = tf.keras.models.load_model('./saved_items/Model16032020.h5')

app = Flask(__name__)

CORS(app)

consumer_key = 'VeF55amirqW1fg08vkpq8F6wm'
consumer_secret = 'njZnUCqgm46eSqUotuynctEdpQgTjgsXCWjkGiJLGYsvIHWugX'
access_token = '1051095766377222145-Jr12rzUXEdMRH2jeNF5f92uYgAp8J1'
access_token_secret = 'd4u8R4G7voFXnxyGGRrZQyik9UqAAL94SmjlV9g6hl6Kt'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


def get_replies(name, tweet_id):
    replies = []
    for tweet in tweepy.Cursor(api.search, q='to:'+name, result_type='recent', timeout=999999).items(1000):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str == tweet_id):
                replies.append(tweet.text)
    return replies


def bytesToJson(byteVal):
    strVal = byteVal.decode("utf-8").replace("'", '"')
    jsonVal = json.loads(strVal)
    return jsonVal


@app.route('/search', methods=['POST'])
def searchUser():
    data = bytesToJson(request.data)['data']
    url = data['tweet_url']
    url = url.replace('https://twitter.com/', '')
    url = url.replace('/status/', '|')
    name, user_id = url.split('|')
    tweet = api.get_status(user_id, tweet_mode='extended')
    user_details = tweet.user
    details = {'name': user_details.name, 'screen_name': user_details.screen_name,
               'description': user_details.description, 'profile_pic': user_details.profile_image_url_https}
    tweet = tweet.full_text
    replies = get_replies(name, user_id)
    print("Replies Extracted!")
    classes = model.predict_classes(prepare_replies(replies)).flatten()
    print("Predictions Done!")
    return jsonify({'replies': replies, 'user_details': details, 'tweet': tweet, 'classes': classes.tolist(), 'count': {'replies': len(classes), 'positive': int(sum(classes)), 'negative': int(len(classes)-sum(classes))}})


if __name__ == "__main__":
    app.run(debug=True)
