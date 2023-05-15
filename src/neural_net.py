# system tools
import os
import cv2
import sys
sys.path.append(".")
import numpy as np
# data munging tools
import pandas as pd
# data loader
from tensorflow.keras.datasets import cifar10
# machine learning tools
#from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# classification models
from sklearn.neural_network import MLPClassifier
# saving models 
from joblib import dump
# import my functions in the utils folder preprocessing script 
from utils.preprocessing import load_data
from utils.preprocessing import preprocess




# neural net classifier 
def neural_net_clf(X_train_dataset, y_train, X_test_dataset, y_test):
    clf = MLPClassifier(random_state=42,
                        hidden_layer_sizes=(64, 10), 
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        max_iter=20).fit(X_train_dataset, y_train)
    # getting predictions 
    y_pred = clf.predict(X_test_dataset)
    return clf, y_pred



def main():
    # load and label the data
    X_train, y_train, X_test, y_test, labels = load_data()
    # preprocess the data
    X_train_dataset, X_test_dataset = preprocess(X_train, X_test)
    # applying the neural net classifier 
    clf, y_pred = neural_net_clf(X_train_dataset, y_train, X_test_dataset, y_test)
    # getting the report 
    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels)
    # save the report 
    with open(os.path.join("out", "NN_classification_report.txt"), "w") as f:
        f.write(report)
     # save the model
    dump(clf, os.path.join("models", "NeuralNetClassifier.joblib"))

if __name__=="__main__":
    main()