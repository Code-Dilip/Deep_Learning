import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import h5py
    
print(os.getcwd())   
    
def load_dataset():
    train_dataset = h5py.File('/home/code_dilip/AI_and_ML/Neural_Network_Intro/coursera-deep-learning-specialization-master/C1 - Neural Networks and Deep Learning/Week 2/Logistic Regression as a Neural Network/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/home/code_dilip/AI_and_ML/Neural_Network_Intro/coursera-deep-learning-specialization-master/C1 - Neural Networks and Deep Learning/Week 2/Logistic Regression as a Neural Network/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

