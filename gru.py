from preprocc import *
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


def gru(train_data, test_data, y_train, y_test, batch_size):
    NUM_WORDS = get_num_words()
    print(NUM_WORDS)
    ## Network architecture
    # inspired at https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e and https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
    model = Sequential()

    #first layer is embedding, takes in size of vocab, 100 dim embedding, and 150 which is length of the comment
    model.add(Embedding(NUM_WORDS, 100, input_length=150))
    model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='sigmoid'))#change to 6
    model.summary() #Print model Summary
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='gru_model_plot.png', show_shapes=True, show_layer_names=True)

    #first run through didn't specify a batch size, probably do that
    #on the next try.
    history = model.fit(train_data, np.array(y_train), validation_split=.2, epochs=3, batch_size=batch_size)
    
    #save json model
    gru_model = model.to_json()
    with open("models/gru.json", "w") as json_file:
        json_file.write(gru_model)

    # serialize weights to HDF5
    model.save_weights("models/gru.h5")
    print("Saved gru to disk")

    score = model.evaluate(test_data, y_test, verbose=1)
    for i in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (model.metrics_names[i], score[i]*100))

    return model, history 


print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("preprocessing...")
X_train, X_test = preprocess(X_train, X_test, False)
# X_train, X_test = preprocess_cached()
print("tokenization...")
train_data, test_data = tokenize(X_train, X_test)
print(type(train_data))
print("model!!!")
model, history = gru(train_data, test_data, y_train, y_test, batch_size=20)
# print("evaluate!!")
# score, history = eight_way_eval(model, test_data, y_test)
# print(score)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print("DONE!")
