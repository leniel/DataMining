import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

path = os.path.realpath('..')

# Loading the data used to train
trainingSet = pd.read_csv(os.path.join(
    path, '../Data/classification-training.csv'), sep=',', header=None)

classes = trainingSet[trainingSet.columns[22]]  # Last column
# Columns between indexes 1 to 22
features = trainingSet[trainingSet.columns[1:22]]

#pd.set_option('display.max_columns', 23)
# print(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, classes, test_size=0.3, random_state=0)

print("(Rows, Columns)" + X_train.shape, y_train.shape)
print("(Rows, Columns)" + X_test.shape, y_test.shape)

classifier = GaussianNB().fit(X_train, y_train)

print(classifier.score(X_test, y_test))
