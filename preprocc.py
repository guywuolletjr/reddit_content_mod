import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import string
import datetime
import re


#modeling imports
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf


import re
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 


#tf imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

#multinomial nb
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from scipy.sparse.linalg import svds

import re
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
#lemmatizing
from nltk.stem import WordNetLemmatizer 

BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)

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


#input is X_train and X_test comment dataframe
#COLUMN WITH DATA NEEDS TO BE NAMED COMMENT_TEXT
def preprocess(X_train, X_test):
    #convert text to lowercase
    X_train = X_train.apply(lambda x: x.astype(str).str.lower())
    X_test = X_test.apply(lambda x: x.astype(str).str.lower())

    #column now has the expanded contractions
    X_train['expanded'] = X_train.comment_text.apply(expandContractions)
    X_test['expanded'] = X_test.comment_text.apply(expandContractions)

    #remove numbers, punctuation
    #https://medium.com/@chaimgluck1/have-messy-text-data-clean-it-with-simple-lambda-functions-645918fcc2fc
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('','','1234567890')))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('','','1234567890')))

    #take away the \n vals
    X_train['expanded'] = X_train.expanded.apply(lambda x: x.translate(str.maketrans('\n',' ')))
    X_test['expanded'] = X_test.expanded.apply(lambda x: x.translate(str.maketrans('\n',' ')))

    #remove english stop words
    X_train['no_stop'] = X_train['expanded'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOPWORDS)]))
    X_test['no_stop'] = X_test['expanded'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOPWORDS)]))

    #lemmatization
    X_train['text'] = X_train.no_stop.apply(lemmatize_text)
    X_test['text'] = X_test.no_stop.apply(lemmatize_text)

    
    return X_train, X_test


