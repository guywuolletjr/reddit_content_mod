import click
import numpy as np
import utils


@cli.command()
@click.option('--Cs', default=10,  help="How many Cs values are chosen in a grid for regularization")
@click.option('--cv', default=5,  help="Specifies K for stratified k fold cross validation")
@click.option('--penalty', default="l2",  help="l2|l1, regularization parameter")
@click.option('--scoring', default="accuracy",  help="Any function in sk.metrics, some options are accuracy, f1, f1_micro, f1_macro,precision, recall, r2 ")
@click.option('--max_iter', default=10000,  help="max iterations till convergence")
@click.option('--ngrams', default=1,  help="Larges ngram value to consider in tf-idf vectorization")
@click.option('--count', is_flag=True, default=False,  help="uses a plain count vectorizer instead of a tf-idf vectorizer")
def tfidf(Cs, cv, penalty, scoring, max_iter, ngrams, count):
    """This function runs a TF-IDF baseline with linear regression.

    References:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    
