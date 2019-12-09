import sys, os
import numpy as np
import pandas as pd
import string
import re

from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')

#tf imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

#keras
from keras import Sequential
import keras
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences #padding
from keras.utils.vis_utils import plot_model #visualisation

import re
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
#lemmatizing
from nltk.stem import WordNetLemmatizer


# ##AUGMENTING STUFF
# from aug_functions import *
# from snorkel.augmentation import RandomPolicy
# from snorkel.augmentation import MeanFieldPolicy
# from snorkel.augmentation import PandasTFApplier

BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
NUM_WORDS = 0

CLIST = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}



#HELPER FUNCTION to the preprocessing
def expandContractions(text):
    c_re = re.compile('(%s)' % '|'.join(CLIST.keys()))

    def replace(match):
        return CLIST[match.group(0)]
    return c_re.sub(replace, text)


#loads the data from the train.csv into X_train etc.
def load_data():
    data = os.path.join("data", "train.csv")

    df = pd.read_csv(data)
    X_train = df[['comment_text']]
    y_train = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    X_test = pd.read_csv(os.path.join("data", "test.csv"))
    y_test = pd.read_csv(os.path.join("data", "test_labels.csv"))
    test = X_test.merge(y_test, on='id')
    test = test[ (test['toxic']!=-1) | (test['severe_toxic']!=-1) |
                (test['obscene']!=-1) | (test['threat']!=-1) | (test['insult']!=-1)
                | (test['identity_hate']!=-1) ]
    test = test.reset_index(drop=True)

    X_test = test[['comment_text']]
    y_test = test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


    return X_train, y_train, X_test, y_test


#lemmatizer helper
def lemmatize_text(text):
#     nltk.download('wordnet')
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    tokens =  [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    proc = ' '.join(tokens)
    return proc

def preprocess_cached():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")

    return X_train, X_test



#input is X_train and X_test comment dataframe
#COLUMN WITH DATA NEEDS TO BE NAMED comment_text
#CHANGE FILENAME if working with more than one dataset
def preprocess(X_train, X_test, augment=False):
    #convert text to lowercase
    print("lowercase")
    X_train['lower'] = X_train.comment_text.apply(lambda x: x.lower())
    X_test['lower'] = X_test.comment_text.apply(lambda x: x.lower())

    #column now has the expanded contractions
    print("contractions")
    X_train['expanded'] = X_train.lower.apply(expandContractions)
    X_test['expanded'] = X_test.lower.apply(expandContractions)

    #remove numbers, punctuation
    #https://medium.com/@chaimgluck1/have-messy-text-data-clean-it-with-simple-lambda-functions-645918fcc2fc
    print("numbers")
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('','','1234567890')))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('','','1234567890')))

    #take away the \n vals
    print("newlines")
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('\n',' ')))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('\n',' ')))

    #remove english stop words
    print("stopwords")
    X_train['no_stop'] = X_train['expanded'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOPWORDS)]))
    X_test['no_stop'] = X_test['expanded'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOPWORDS)]))

    #BEFORE THE LEMMATIZATION WE HAVE TO DO THE AUGMENTATION
    if(augment):
        X_train = augment(X_train, y_train)


    #lemmatization
    print("lemmas")
    X_train['text'] = X_train.no_stop.apply(lemmatize_text)
    X_test['text'] = X_test.no_stop.apply(lemmatize_text)

    export_train = X_train.to_csv('data/X_train.csv', index = None, header=True)
    export_test = X_test.to_csv('data/X_test.csv', index = None, header=True)

    return X_train, X_test

# #need to
# def load_data_from_csv():
#     X_train = pd.read_csv("X_train.csv")
#     X_test = pd.read_csv("X_test.csv")
#     y_train = pd.read_csv("y_train.csv")
#     y_test  = pd.read_csv("y_test.csv")

#     return X_train, X_test, y_train, y_test

def tokenize_baseline(X_train, X_test):
    #train tokenizer, then encode documents (comment_text)
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(X_train['comment_text']) #fit the tokenizer to training set, make the test unknown words be an UNK value.
    global NUM_WORDS
    NUM_WORDS = len(tokenizer.word_index) + 1
    print(NUM_WORDS, "first!!!")

    train_sequences = tokenizer.texts_to_sequences(X_train['comment_text'])
    train_data = pad_sequences(train_sequences, maxlen=150)
    test_sequences = tokenizer.texts_to_sequences(X_test['comment_text'])
    test_data = pad_sequences(test_sequences, maxlen=150)


    return train_data, test_data


def tokenize(X_train, X_test):
    #train tokenizer, then encode documents (comment_text)
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(X_train['text']) #fit the tokenizer to training set, make the test unknown words be an UNK value.
    global NUM_WORDS
    NUM_WORDS = len(tokenizer.word_index) + 1
    print(NUM_WORDS, "first!!!")

    train_sequences = tokenizer.texts_to_sequences(X_train['comment_text'])
    train_data = pad_sequences(train_sequences, maxlen=150)
    test_sequences = tokenizer.texts_to_sequences(X_test['comment_text'])
    test_data = pad_sequences(test_sequences, maxlen=150)


    return train_data, test_data


def get_num_words():
    global NUM_WORDS
    print("RETURNING THIS:", NUM_WORDS)
    return NUM_WORDS

def eight_way(train_data, test_data, y_train, y_test, batch_size):
    global NUM_WORDS
    print(NUM_WORDS)
    ## Network architecture
    # inspired at https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e and https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
    model = Sequential()

    #first layer is embedding, takes in size of vocab, 100 dim embedding, and 150 which is length of the comment
    model.add(Embedding(NUM_WORDS, 100, input_length=150))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='sigmoid'))#change to 6
    model.summary() #Print model Summary
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='lstm_model_plot.png', show_shapes=True, show_layer_names=True)
    print("done")
    #first run through didn't specify a batch size, probably do that
    #on the next try.
    model.fit(train_data, np.array(y_train), validation_split=.2, epochs=3, batch_size=batch_size)

    #save json model
    eight_way_json = model.to_json()
    with open("eight_way.json", "w") as json_file:
        json_file.write(eight_way_json)

    # serialize weights to HDF5
    model.save_weights("eight_way.h5")
    print("Saved eight_way to disk")

    return model

def print_results(model, X, y):
    """Print out evaluate a model, returns the metrics as tuple
    """
    prediction = model.predict(X)
    precision, recall, fbeta_score, support = \
        precision_recall_fscore_support(y, prediction)
    accuracy = accuracy_score(y, prediction)

    print ("Precision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\nAccuracy".format(
            precision, recall, fbeta_score, support))

    print (classification_report(y, prediction))

    print("Example predictions: ")
    print(prediction)

    return (precision, recall, fbeta_score, support, accuracy)

def eight_way_eval(model, test_data, y_test):
    score = model.evaluate(test_data, y_test)
    print_results(model, test_data, y_test)
    return score

def model_eval(model, test_data, y_test):
    score = model.evaluate(test_data, y_test)
    print_results(model, test_data, y_test)
    return score
