from preprocc import *

print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("preprocessing...")
X_train, X_test = preprocess(X_train, X_test, False)
print("tokenization...")
train_data, test_data = tokenize(X_train, X_test)
print("model!!!")
model = eight_way(train_data, test_data, y_train, y_test, batch_size=20)
print("evaluate!!")
score = eight_way_eval(model, test_data, y_test)
print(score)
print("DONE!")

