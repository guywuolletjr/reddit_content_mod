from preprocc import *

print("loading data...")
X_train, y_train, X_test, y_test = load_data()
print("preprocessing...")
X_train, X_test = preprocess(X_train, X_test, augment=False) #DIFFERENCE IS TRUE for augmentation

