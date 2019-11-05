import sys, os
import pandas as pd


def load_data():

    train_data = os.path.join("data", "train.csv")

    df = pd.read_csv(data)
    X_train = df['id', 'comment_text']
    y_train = df['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    X_test = os.path.join("data", "test.csv")
    y_test = os.path.join("data", "test_labels.csv")

    return X_train, y_train, X_test, y_test 
