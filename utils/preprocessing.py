# system tools
import os
import cv2
import numpy as np
# data loader
from tensorflow.keras.datasets import cifar10


def load_data(): 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    return X_train, y_train, X_test, y_test, labels


def preprocess(X_train, X_test):
    # gray scaling each image in the test and train sets 
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # rescaling 
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # reshaping the train dataset 
    nsamples, nx, ny = X_train_scaled.shape 
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny)) 
    # reshaping the test dataset 
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train_dataset, X_test_dataset

