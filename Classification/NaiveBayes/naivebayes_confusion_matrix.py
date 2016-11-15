import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

path = os.path.realpath('..')

# Used to load external functions...
sys.path.append(path)

from utils import plot_confusion_matrix

# Loading the data used to train
trainingSet = pd.read_csv(os.path.join(
    path, '../Data/classification-training.csv'), sep=',', header='infer')

classes = trainingSet[trainingSet.columns[13]]  # Last column

# Columns between indexes 1 to 22
features = trainingSet[trainingSet.columns[1:12]]

#pd.set_option('display.max_columns', 23)
# print(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, classes, test_size=0.3, random_state=0)

#print("(Rows, Columns)" + X_train.shape, y_train.shape)
#print("(Rows, Columns)" + X_test.shape, y_test.shape)

predictions = GaussianNB().fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)

# Get only distinc values
classes = list(set(classes))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
