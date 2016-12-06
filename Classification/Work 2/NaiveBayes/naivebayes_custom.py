'''
 Created on Sat Nov 05 2016

 Copyright (c) 2016 Leniel Macaferi's Consulting
'''

# Adapted from:
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python

import csv
import math
import os
import random
import numpy as np
import pandas as pd
import string


def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / \
        float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]

    del summaries[0]  # This is the id
    del summaries[-1]  # This is the class\label

    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():

    path = os.path.realpath('..')

    trainingSet = loadCsv(os.path.join(
        path, '../Data/classification-training.csv'))

    # Prepare model
    summaries = summarizeByClass(trainingSet)

    testData = pd.read_csv(os.path.join(
        path, '../Data/classification-test.csv'), header=None, chunksize=1)

    ids = []
    predictionSet = []

    for dataFrame in testData:

        ids.append(dataFrame.values[0][0])

        # Skip 1st column (id)
        df = dataFrame[dataFrame.columns.difference([0])]

        predictionSet.append(df.values[0])

    # Test model
    predictions = getPredictions(summaries, predictionSet)

    writer = csv.writer(
        open(os.path.join(path, '../NaiveBayes/nb-predictions.csv'), 'w'))

    alphabet = dict(enumerate(string.ascii_uppercase, 1))
    # print(d[3]) # C

    for prediction, id in zip(predictions, ids):

        data = [id, alphabet[prediction]]
        print(data)

        writer.writerow(data)

    #accuracy = getAccuracy(predictionSet, predictions)

    #print("Accuracy: {0}%".format(accuracy))

    '''
  testSet = pd.read_csv(os.path.join(path, 'Data/classification-test.csv'), sep=',', header = None, chunksize=1)
 
  for df in testSet:
    df2 = df[df.columns.difference([0])] # Skip 1st column (id)

    inputVector = df2.values[0]

    prediction = predict(summaries, inputVector)

    print("Prediction: for id = {0} is class\label = {1}\n".format(df.values[0][0], prediction))
  '''

main()
