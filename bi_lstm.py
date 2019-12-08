from preprocc import *

def bi_lstm(train_data, test_data, y_train, y_test, batch_size):
    NUM_WORDS = get_num_words()
    print(NUM_WORDS)
    model = Sequential()

    #first layer is embedding, takes in size of vocab, 100 dim embedding, and 150 which is length of the comment
    model.add(Embedding(NUM_WORDS, 100, input_length=150))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(6, activation='sigmoid'))#change to 6
    model.summary() #Print model Summary
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    plot_model(model, to_file='bi_lstm_model_plot.png', show_shapes=True, show_layer_names=True)
    #first run through didn't specify a batch size, probably do that
    #on the next try.
    model.fit(train_data, np.array(y_train), validation_split=.2, epochs=3, batch_size=batch_size)

    #save json model
    bi_lstm_model = model.to_json()
    with open("models/bi_lstm.json", "w") as json_file:
        json_file.write(bi_lstm_model)

    # serialize weights to HDF5
    model.save_weights("models/bi_lstm.h5")
    print("Saved bidirectional lstm to disk")

    return model

print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("preprocessing...")
X_train, X_test = preprocess(X_train, X_test) #DIFFERENCE IS TRUE for augmentation
print("tokenization...")
train_data, test_data = tokenize(X_train, X_test)
num_words = get_num_words()
print("model!!!")
model = bi_lstm(train_data, test_data, y_train, y_test, batch_size=20)
print("evaluate!!")
score = eight_way_eval(model, test_data, y_test)
print(score)
print("DONE!")
