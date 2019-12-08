import sys, os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


def load_data():

    data = os.path.join("data", "train.csv")

    df = pd.read_csv(data)
    X_train = df[['id', 'comment_text']]
    y_train = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    X_test = pd.read_csv(os.path.join("data", "test.csv"))
    y_test = pd.read_csv(os.path.join("data", "test_labels.csv"))
    test = X_test.merge(y_test, on='id')
    test = test[ (test['toxic']!=-1) | (test['severe_toxic']!=-1) |
                (test['obscene']!=-1) | (test['threat']!=-1) | (test['insult']!=-1)
                | (test['identity_hate']!=-1) ]
    test = test.reset_index(drop=True)

    X_test = test[['id', 'comment_text']]
    y_test = test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    return X_train, y_train, X_test, y_test
    
def load_reddit_data(split=.9):
    
    mod_df = pd.read_csv('data/reddit/reddit-removal-log.csv')
    unmod_df = pd.read_csv('data/reddit/data2M.csv')
    
    mod_df['moderated'] = 1
    unmod_df['moderated'] = 0
    
    df = pd.concat([mod_df, unmod_df])
    df = df.reset_index(drop=True)
    
    X = df[['body', 'subreddit']]
    y = df[['moderated']]
    
    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
    return X_train, X_test, y_train, y_test

def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for i, row in df.sample(frac=1, random_state=2).iterrows():
            transformed_or_none = f(row)
            # If TF returned a transformed example, record it in dict and move to next TF.
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            "TF Name": f.name,
                            "Original Text": row.text,
                            "Transformed Text": transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)

