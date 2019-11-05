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
    """
    This function runs a TF-IDF baseline with linear regression.

    Documentation
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    def tfidf_transform(corpus, is_train=False):
        if is_train:
            X = vectorizer.fit_transform(corpus)
        else:
            X = vectorizer.transform(corpus)
        return X

    def print_results(model, X, y):
        """
        Evaluate the model, prints results, and return as tuple
        """
        prediction = model.predict(X)
        precision, recall, fbeta_score, support = \
            precision_recall_fscore_support(y, prediction)
        accuracy = accuracy_score(y, prediction)

        print ("Precision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\nAccuracy {}\n".format(
                precision, recall, fbeta_score, support, accuracy))

        print (classification_report(y, prediction))

        return (precision, recall, fbeta_score, support, accuracy)


    X_train, y_train, X_test, y_test = load_data()
    np.random.seed(42) #so that our results are the same each time we run

    if(ngrams > 3):
        ngrams = 3 # this makes training not take forever

    vectorizer = TfidfVectorizer(ngram_range=(1, ngrams))
    if(count):
        vectorizer = CountVectorizer()

    # turn the corpus into a list to pass into the vectorizer
    train_corpus = X_train['comment_text'].values.tolist()
    test_corpus = X_test['comment_text'].values.tolist()

    X_train_tfidf = tfidf_transform(train_corpus, is_train=True)
    X_test_tfidf = tfidf_transform(test_corpus, is_train=False)


    labels = ['toxic' , 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    accuracies = []

    for label in labels:
        print("Train model on the \"{}\" label.\n".format(label))
        y_train_label = y_train[label].values.tolist()
        y_test_label = y_test[label].values.tolist()

        lr = LogisticRegressionCV(   class_weight="balanced",
                                     Cs = Cs,
                                     cv = cv,
                                     penalty = penalty,
                                     scoring = scoring,
                                     max_iter = max_iter)

        lr.fit(X_train_tfidf, y_train_label)

        print("Train Results\n")
        print_results(lr, X_train_tfidf, y_train_label)

        print("Test Results\n")
        precision, recall, fbeta_score, support, accuracy = print_results(lr, X_test_tfidf, y_test_label)
        accuracies.append(accuracy)

    macro_accuracy = np.average(np.array(accuracies))
    print("FINAL RESULT\n Macro Accuracy averaged across all labels = {}".format(macro_accuracy))
