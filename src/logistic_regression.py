# system tools
import os, sys
sys.path.append(".")
import cv2
import numpy as np
# data loader
from tensorflow.keras.datasets import cifar10
# data munging tools
import pandas as pd
# machine learning tools
from sklearn.metrics import classification_report
# classification models
from sklearn.linear_model import LogisticRegression
# saving models 
from joblib import dump
# import my functions in the utils folder preprocessing script 
bash
from utils.preprocessing import preprocess


# log reg classifier
def log_reg_clf(X_train_dataset, y_train, X_test_dataset, y_test):
    clf = LogisticRegression(penalty="none", 
                            tol=0.1, 
                            verbose=True, 
                            solver="saga",
                            multi_class="multinomial").fit(X_train_dataset, y_train)
    # getting predictions based on the model 
    y_pred = clf.predict(X_test_dataset)
    return clf, y_pred 


def main():
    # load and label the data
    X_train, y_train, X_test, y_test, labels = load_data()
    # preprocess the data
    X_train_dataset, X_test_dataset = preprocess(X_train, X_test)
    # applying the logistic regression classifier 
    clf, y_pred = log_reg_clf(X_train_dataset, y_train, X_test_dataset, y_test)
    # getting the report 
    report = classification_report(y_test, 
                                y_pred, 
                                target_names=labels)
    # save the report 
    with open(os.path.join("out", "logistic_classification_report.txt"), "w") as f:
        f.write(report)
    # save the model
    dump(clf, os.path.join("models", "LogisticClassifier.joblib"))


if __name__=="__main__":
    main()



