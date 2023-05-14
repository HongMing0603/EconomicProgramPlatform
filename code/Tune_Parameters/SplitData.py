from sklearn.model_selection import train_test_split
import pandas as pd
import sys

def data_split(X, y):
    """
    X: array-like
    y: array-like
    """
    while True:
        try:
            # test_size = float(input("Please enter your test size(Like 0.2):"))
            test_size = float(0.2)
            X_train, X_test, y_train, y_test = trainTest_split(X, y, testSize=test_size)
            return X_train, X_test, y_train, y_test
        except:
            print("Input error")
            

def trainTest_split(X, y, testSize=0.2, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=shuffle)
    return X_train, X_test, y_train, y_test

def data_split_AR(Data):
    """
    AR series dedicated data segmentation
    """
    while True:
        try:
            test_size_input = float(input("Please enter your test size(Like 0.2):"))
            # Determine if it is a decimal point(No more than 1)
            if test_size_input <= 1.0:
                test_size = int(len(Data)*test_size_input)
                train_data, test_data = Data.iloc[ : -test_size, : ], Data.iloc[-test_size : , : ]
                return train_data, test_data
            else:
                print("Input error")
        except:
            print("Input error")
