import sys
import pickle
import re
import tensorflow as tf
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_data(data,lemmatize=True,save=True):
    
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    #Internal function
    def __internal_preprocessor(text):
        #Remove links and user mentions
        text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
        
        #Convert to lowercase
        text = text.lower()
        
        #Remove Punctuation
        text = text.translate(str.maketrans('', '',punctuation))
        
        #Lemmatization
        if lemmatize:
            text = ' '.join([ lemmatizer.lemmatize(word) for word in text.split() ])
            
        return text
    
    #Process all reviews
    processed_data = list(map(__internal_preprocessor,data))
    
    #Save the processed data
    if save:
        path = r'./saved_items/processed_reviews.data'
        with open(path, 'wb') as f:
            # store the data as binary data stream
            pickle.dump(processed_data, f)
        print("Saved data at:",path)
        
    return processed_data   

def build_vocabulary(data_list):
    unique_words = set(' '.join(data_list).split(" "))
    vocab = { k:v for v,k in enumerate(unique_words)}
    return vocab

def encode_pad_data(data, vocab):
    return pad_sequences(list(map(lambda entry: [vocab[word] for word in entry.split()] ,data)),
                                  padding="post")

def prepare_replies(replies):
    processed_replies = preprocess_data(replies,save=False)
    vocab = build_vocabulary(processed_replies)
    encoded_replies = encode_pad_data(processed_replies, vocab)
    return encoded_replies