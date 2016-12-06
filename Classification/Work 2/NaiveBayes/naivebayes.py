'''
 Created on Sat Nov 05 2016

 Copyright (c) 2016 Leniel Macaferi's Consulting
'''

import csv
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import string

path = os.path.realpath('..')

# Loading the Training set...
trainingSet = pd.read_csv(os.path.join(
    path, '../Data/classification-training.csv'), sep=',', header=None)

classes = trainingSet[trainingSet.columns[22]]  # Last column
# Columns between indexes 1 to 21
features = trainingSet[trainingSet.columns[1:22]]

#pd.set_option('display.max_columns', 23)
# print(features)

# Gaussian Naive Bayes algorithm
classifier = GaussianNB(priors=None)

classifier.fit(features, classes)

# Loading the Test set...
testSet = pd.read_csv(os.path.join(
    path, '../Data/classification-test.csv'), sep=',', header=None)

# Getting the ids that are used only to output the result
ids = testSet[testSet.columns[0]]

# Using the trained classifier to predict the test data
predictions = classifier.predict(testSet[testSet.columns[1:22]])

# Write to save predictions to disk
writer = csv.writer(open(os.path.join(path, 'nb-predictions.csv'), 'w'))

for prediction, id in zip(predictions, ids):
    data = [id, prediction]

    #print("{0} = {1}\n".format(id, prediction))

    writer.writerow(data)
