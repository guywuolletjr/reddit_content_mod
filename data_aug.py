import sys, os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
import string
import datetime
import re
import click


from snorkel.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

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

@click.group()
def cli():
    pass

@cli.command()
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

@cli.command()
#@transformation_function(pre=[spacy])
def change_person(x):
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
    # If there is at least one person name, replace a random one. Else return None.
    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.text = x.text.replace(name_to_replace, replacement_name)
        return x


# Swap two adjectives at random.
@cli.command()
#@transformation_function(pre=[spacy])
def swap_adjectives(x):
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap.
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.text = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x

    
@cli.command()
def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")

@cli.command()
def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])

@cli.command()
#@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    # Get indices of verb tokens in sentence.
    verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if verb_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(verb_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="v")
        # If there's a valid verb synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x

@cli.command()
#@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    # Get indices of noun tokens in sentence.
    noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if noun_idxs:
        # Pick random noun idx to replace.
        idx = np.random.choice(noun_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="n")
        # If there's a valid noun synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x

@cli.command()
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
            x.text = replace_token(x.doc, idx, synonym)
            return x

tfs = [
    change_person,
    swap_adjectives,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    replace_adjective_with_synonym,
]

from utils import preview_tfs

preview_tfs(X_train['expanded'], tfs)

#random_policy = RandomPolicy(
    #len(tfs), sequence_length=2, n_per_original=2, keep_original=True
#)

#mean_field_policy = MeanFieldPolicy(
    #len(tfs),
    #sequence_length=2,
    #n_per_original=2,
    #keep_original=True,
    #p=[0.05, 0.05, 0.3, 0.3, 0.3],
#)

#tf_applier = PandasTFApplier(tfs, mean_field_policy)
#df_train_augmented = tf_applier.apply(train)
#Y_train_augmented = df_train_augmented["label"].values

#print(f"Original training set size: {len(train)}")
#print(f"Augmented training set size: {len(df_train_augmented)}")
