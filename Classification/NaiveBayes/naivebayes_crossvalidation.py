'''
 Created on Sat Nov 05 2016

 Copyright (c) 2016 Leniel Macaferi's Consulting
'''

import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

path = os.path.realpath('..')

# Loading the data used to train
trainingSet = pd.read_csv(os.path.join(path, '../Data/classification-training.csv'), sep=',', header = None)

classes = trainingSet[trainingSet.columns[22]] # Last column
features = trainingSet[trainingSet.columns[1:22]] # Columns between indexes 1 to 22

#pd.set_option('display.max_columns', 23)
#print(features)

classifier = GaussianNB()

scores = cross_val_score(classifier, features, classes, cv = 5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))