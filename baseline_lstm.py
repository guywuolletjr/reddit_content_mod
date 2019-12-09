from preprocc import *

def baseline_lstm(train_data, test_data, y_train, y_test, batch_size):
    global NUM_WORDS
    print(NUM_WORDS)
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
    baseline_model = model.to_json()
    with open("models/baseline_lstm.json", "w") as json_file:
        json_file.write(baseline_model)

    # serialize weights to HDF5
    model.save_weights("models/baseline_lstm.h5")
    print("Saved baseline_lstm to disk")

    score = model.evaluate(test_data, y_test, verbose=1)
    for i in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (model.metrics_names[i], score[i]*100))

    return model, score 


print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("tokenization...")
train_data, test_data = tokenize_baseline(X_train, X_test)
print(type(train_data))
print("model!!!")
model, score = baseline_lstm(train_data, test_data, y_train, y_test, batch_size=20)
print(score)


print("DONE!")