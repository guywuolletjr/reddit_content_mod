import click
import numpy as np
import utils


@cli.command()
@click.option('--penalty', default="l2",  help="l2|l1, regularization parameter")
@click.option('--max_iter', default=10000,  help="max iterations till convergence")
@click.option('--num_c', default=10,  help="number of regularization values to try between 1e-4 and 1e4")
@click.option('--scoring_function', default="f1_macro",  help="accuracy|f1|f1_micro|f1_macro|precision|recall|r2 (or any function defined by sklearn.metrics)")
@click.option('--k_folds', default=5,  help="number of folds for k-fold cross validation")
@click.option('--max_ngrams', default=1,  help="maximum number of ngrams to consider in tf-idf vectorization (default 1)")
@click.option('--augmentation', is_flag=True, default=False,  help="whether or not to do data augmentation (see data_utils.py)")
@click.option('--net', is_flag=True, default=False, help='whether or not to run the neural net instead of tfidf')
@click.option('--count', is_flag=True, default=False,  help="uses a plain count vectorizer instead of a tf-idf vectorizer")
@click.option('--epoch', default=100, help='how many epoch to ball for')
def tfidf(penalty, max_iter, num_c,
          scoring_function, k_folds,
          max_ngrams, augmentation, net, count,
          epoch):
    """Runs a TF-IDF and logistic regression baseline.

    Usage: `python baseline.py tfidf`

    References:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    pass
