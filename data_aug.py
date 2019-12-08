import sys, os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
import string
import datetime
import re
#import click


from snorkel.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="expanded", doc_field="doc", memoize=True)

import names
from snorkel.augmentation import transformation_function

replacement_names = [names.get_full_name() for _ in range(50)]

import nltk
import ssl
from nltk.corpus import wordnet as wn

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download("wordnet")

from snorkel.augmentation import RandomPolicy
from snorkel.augmentation import MeanFieldPolicy
from snorkel.augmentation import PandasTFApplier

#@click.group()
#def cli():
   # pass

#@cli.command()
def dummy():
    """
    Usage: `python main.py dummy`
    """
    raise NotImplementedError(
        "dont actually run this"
    )

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

X_train, y_train, X_test, y_test = load_data()

import re
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
#lemmatizing

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)

cList = {
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

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

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

#@cli.command()
@transformation_function(pre=[spacy])
def change_person(x):
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
    # If there is at least one person name, replace a random one. Else return None.
    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.expanded = x.expanded.replace(name_to_replace, replacement_name)
        return x


# Swap two adjectives at random.
#@cli.command()
@transformation_function(pre=[spacy])
def swap_adjectives(x):
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap.
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.expanded = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x

    
#@cli.command()
def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")

#@cli.command()
def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])

#@cli.command()
@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    # Get indices of verb tokens in sentence.
    verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if verb_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(verb_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="v")
        # If there's a valid verb synonym, replace it. Otherwise, return None.
        if synonym:
            x.expanded = replace_token(x.doc, idx, synonym)
            return x

#@cli.command()
@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    # Get indices of noun tokens in sentence.
    noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if noun_idxs:
        # Pick random noun idx to replace.
        idx = np.random.choice(noun_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="n")
        # If there's a valid noun synonym, replace it. Otherwise, return None.
        if synonym:
            x.expanded = replace_token(x.doc, idx, synonym)
            return x

#@cli.command()
#@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
    # Get indices of adjective tokens in sentence.
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    if adjective_idxs:
        # Pick random adjective idx to replace.
        idx = np.random.choice(adjective_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="a")
        # If there's a valid adjective synonym, replace it. Otherwise, return None.
        if synonym:
            x.expanded = replace_token(x.doc, idx, synonym)
            return x

tfs = [
    change_person,
    swap_adjectives,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    #replace_adjective_with_synonym,
]

from utils import preview_tfs
prev = X_train['expanded'].to_frame()
#print(prev.head())

#preview_tfs(prev, tfs)
#print(preview_tfs(X_train, tfs))

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True
)

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=4,
    keep_original=True,
    p=[0.1, 0.1, 0.4, 0.4],
)

tf_applier = PandasTFApplier(tfs, mean_field_policy)
df_train_augmented = tf_applier.apply(X_train)
Y_train_augmented = df_train_augmented["label"].values

print(f"Original training set size: {len(X_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")
