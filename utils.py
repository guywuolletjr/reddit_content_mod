import sys, os
import pandas as pd


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
