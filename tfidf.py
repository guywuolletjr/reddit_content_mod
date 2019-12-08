import click
import numpy as np
import utils

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from preprocc import preprocess

@click.command()
@click.option('--cs', default=10,  help="How many Cs values are chosen in a grid for regularization")
@click.option('--cv', default=5,  help="Specifies K for stratified k fold cross validation")
@click.option('--penalty', default="l2",  help="l2|l1, regularization parameter")
@click.option('--scoring', default="accuracy",  help="Any function in sk.metrics, some options are accuracy, f1, f1_micro, f1_macro,precision, recall, r2 ")
@click.option('--max_iter', default=10000,  help="max iterations till convergence")
@click.option('--ngrams', default=1,  help="Larges ngram value to consider in tf-idf vectorization")
@click.option('--count', is_flag=True, default=False,  help="uses a plain count vectorizer instead of a tf-idf vectorizer")
@click.option('--reddit', is_flag=True, default=False,  help="train and test on the reddit data")
@click.option('--nn', is_flag=True, default=False,  help="train with a neural network instead of a logistic regression")
def tfidf(cs, cv, penalty, scoring, max_iter, ngrams, count, reddit, nn):
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

    if(reddit):
        X_train, y_train, X_test, y_test = utils.load_reddit_data()
        labels = ['moderated']
        text_col = 'text'
    else:
        X_train, y_train, X_test, y_test = utils.load_data()
        labels = ['toxic' , 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        text_col = 'text'

    X_train, X_test = preprocess(X_train, X_test, False)

    np.random.seed(42) #so that our results are the same each time we run

    if(ngrams > 3):
        ngrams = 3 # this makes training not take forever

    vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), max_features=10000)
    if(count):
        vectorizer = CountVectorizer()

    # turn the corpus into a list to pass into the vectorizer
    train_corpus = X_train[text_col].values.tolist()
    test_corpus = X_test[text_col].values.tolist()

    X_train_tfidf = tfidf_transform(train_corpus, is_train=True)
    X_test_tfidf = tfidf_transform(test_corpus, is_train=False)

    accuracies = []

    if(nn):
        print("Neural Network")
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.utils.vis_utils import plot_model

        if(reddit):
            output_size = 1
        else:
            output_size = 6

        model = Sequential()
        model.add(Dense(128, activation = 'tanh', input_dim=1/0000))
        model.add(Dense(64, activation = 'tanh'))
        model.add(Dense(32, activation = 'tanh'))
        model.add(Dense(output_size, activation = 'sigmoid')) # we have six labels
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        #plot_model(model, to_file='tfidf_model_plot.png', show_shapes=True, show_layer_names=True)

        X_train_tfidf = X_train_tfidf.toarray()
        X_test_tfidf = X_test_tfidf.toarray()
        y_train = y_train.values
        y_test = y_test.values

        model_output = model.fit(X_train_tfidf, y_train, epochs=100, batch_size = 10, validation_split=.3, verbose=1)

        model.save_weights('models/tfidf_nn_weights.h5')
        with open('models/tfidf_nn_architecture.json', 'w') as f:
            f.write(model.to_json())

    else:
        for label in labels:
            print("Train model on the \"{}\" label.\n".format(label))
            y_train_label = y_train[label].values.tolist()
            y_test_label = y_test[label].values.tolist()

            print("Logistic Regression")
            model = LogisticRegressionCV(   class_weight="balanced",
                                            Cs = cs,
                                            cv = cv,
                                            penalty = penalty,
                                            scoring = scoring,
                                            max_iter = max_iter)

            model.fit(X_train_tfidf, y_train_label)

            print("Train Results\n")
            print_results(model, X_train_tfidf, y_train_label)

            print("Test Results\n")
            precision, recall, fbeta_score, support, accuracy = print_results(lr, X_test_tfidf, y_test_label)
            accuracies.append(accuracy)

        macro_accuracy = np.average(np.array(accuracies))
        print("FINAL RESULT\n Macro Accuracy averaged across all labels = {}".format(macro_accuracy))

if __name__ == "__main__":
    tfidf()
