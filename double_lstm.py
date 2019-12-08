from preprocc import *

print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("preprocessing...")
X_train, X_test = preprocess(X_train, X_test) #DIFFERENCE IS TRUE for augmentation
print("tokenization...")
train_data, test_data = tokenize(X_train, X_test)
num_words = get_num_words()
print("model!!!")
model = lstm_double(train_data, test_data, y_train, y_test, batch_size=20, num_words)
print("evaluate!!")
score = eight_way_eval(model, test_data, y_test)
print(score)
print("DONE!")

def lstm_double(train_data, test_data, y_train, y_test, batch_size, NUM_WORDS):
    model = Sequential()

    #first layer is embedding, takes in size of vocab, 100 dim embedding, and 150 which is length of the comment
    model.add(Embedding(NUM_WORDS, 100, input_length=150))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='sigmoid'))#change to 6 
    model.summary() #Print model Summary
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    #first run through didn't specify a batch size, probably do that
    #on the next try.
    model.fit(train_data, np.array(y_train), validation_split=.2, epochs=3, batch_size=batch_size)

    #save json model
    eight_way_json = model.to_json()
    with open("eight_way.json", "w") as json_file:
        json_file.write(eight_way_json)

    # serialize weights to HDF5
    model.save_weights("eight_way.h5")
    print("Saved eight_way to disk")

    return model

def print_results(model, X, y):
    """Print out evaluate a model, returns the metrics as tuple
    """
    prediction = model.predict(X)
    precision, recall, fbeta_score, support = \
        precision_recall_fscore_support(y, prediction)
    accuracy = accuracy_score(y, prediction)

    print ("Precision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\nAccuracy".format(
            precision, recall, fbeta_score, support))

    print (classification_report(y, prediction))

    print("Example predictions: ")
    print(prediction)

    return (precision, recall, fbeta_score, support, accuracy)

def eight_way_eval(model, test_data, y_test):
    score = model.evaluate(test_data, y_test)
    print_results(model, test_data, y_test)
    return score

def model_eval(model, test_data, y_test):
    score = model.evaluate(test_data, y_test)
    print_results(model, test_data, y_test)
    return score
