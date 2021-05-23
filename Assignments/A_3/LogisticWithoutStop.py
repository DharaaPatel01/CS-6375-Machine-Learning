#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 03:30:12 2021

@author: Dhara Patel
"""

import sys
import collections
import os
import re
import codecs
import numpy as np

if (len(sys.argv) != 5):  
    sys.exit("Please give valid Arguments- \n<'train' folder path that has both spam and ham folder> \
              \n<'test' folder path that has both spam and ham folder>\
              \n<Regularization parameter - Lambda value>\
              \n<no. of iterations>")
else:
    train = sys.argv[1]
    test = sys.argv[2]
    Lambda = float(sys.argv[3])
    Iterations = sys.argv[4]

train = "./train"
test = "./test"
# Lambda = 0.0001
# Iteration = 100

count_TrainSpam = 0
count_TrainHam = 0
learningRate = 0.1

stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are",
              "aren't","as","at","be","because","been","before","being","below","between",
              "both","but","by","can't","cannot","could","couldn't","did","didn't","do","does",
              "doesn't","doing","don't","down","during","each","few","for","from","further",
              "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's",
              "her","here","here's","hers","herself","him","himself","his","how","how's","i",
              "i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
              "let's","me","more","most","mustn't","my","myself","no","nor","not","of","off",
              "on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
              "same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
              "such","than","that","that's","the","their","theirs","them","themselves","then",
              "there","there's","these","they","they'd","they'll","they're","they've","this",
              "those","through","to","too","under","until","up","very","was","wasn't","we","we'd",
              "we'll","we're","we've","were","weren't","what","what's","when","when's","where",
              "where's","which","while","who","who's","whom","why","why's","with","won't","would",
              "wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself",
              "yourselves"]

bias = 0
xnode = 1
train_ham_path = train + '/ham'
train_spam_path = train + '/spam'
test_ham_path = test + '/ham'
test_spam_path = test + '/spam'

# Regular expression to clean the data given in train ham and spam folder
regex = re.compile(r'[A-Za-z0-9\']')

def FileOpen(filename, path):
    fileHandler = codecs.open(path + "/" + filename, 'rU',
                              'latin-1')  # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
    words = [Findwords.lower() for Findwords in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    fileHandler.close()
    return words

def browseDirectory(path):
    wordList = list()
    fileCount = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            wordList += FileOpen(files, path)
            fileCount += 1
    return wordList, fileCount

# training vocab
vocab_TrainSpam, count_TrainSpam = browseDirectory(train_spam_path)
vocab_TrainHam, count_TrainHam = browseDirectory(train_ham_path)

# testing vocab
vocab_TestSpam, countTestSpam = browseDirectory(test_spam_path)
vocab_TestHam, countTestHam = browseDirectory(test_ham_path)


def RemoveStopWords():
    for word in stop_words:
        if word in vocab_TrainSpam:
            i = 0
            len_s = len(vocab_TrainSpam)
            while (i < len_s):
                if (vocab_TrainSpam[i] == word):
                    vocab_TrainSpam.remove(word)
                    len_s -= 1
                    continue
                i += 1
        if word in vocab_TrainHam:
            i = 0
            len_h = len(vocab_TrainHam)
            while (i < len_h):
                if (vocab_TrainHam[i] == word):
                    vocab_TrainHam.remove(word)
                    len_h -= 1
                    continue
                i += 1
        if word in vocab_TestSpam:
            i = 0
            len_sTst = len(vocab_TestSpam)
            while (i < len_sTst):
                if (vocab_TestSpam[i] == word):
                    vocab_TestSpam.remove(word)
                    len_sTst -= 1
                    continue
                i += 1
        if word in vocab_TestHam:
            i = 0
            len_hTst = len(vocab_TestHam)
            while (i < len_hTst):
                if (vocab_TestHam[i] == word):
                    vocab_TestHam.remove(word)
                    len_hTst -= 1
                    continue
                i += 1
RemoveStopWords()    


vocab_Train = vocab_TrainHam + vocab_TrainSpam
dict_Train = collections.Counter(vocab_Train)
list_Train = list(dict_Train.keys())
train_TargetList = list()  
total_TrainFiles = count_TrainHam + count_TrainSpam

vocab_Test = vocab_TestHam + vocab_TestSpam
dict_Test = collections.Counter(vocab_Test)
list_Test = list(dict_Test.keys())
test_TargetList = list()
total_TestFiles = countTestHam + countTestSpam

# Initialize the matrix with size(row * column) and assign 0 to each element
def initiliazeMatrix(row, column):
    matrix = [0] * row
    for i in range(row):
        matrix[i] = [0] * column
    return matrix

matrix_train = initiliazeMatrix(total_TrainFiles, len(list_Train))
matrix_test = initiliazeMatrix(total_TestFiles, len(list_Test))

sigmoidList = list()  # for each row
for i in range(total_TrainFiles):
    sigmoidList.append(-1)
    train_TargetList.append(-1)
for i in range(total_TestFiles):
    test_TargetList.append(-1)

weightOfFeature = list()
for feature in range(len(list_Train)):
    weightOfFeature.append(0)

# Fill in values in the matrix for training and testing data
def createMatrix(matrix, path, vocab, row, classifier, FinalList):
    for fileName in os.listdir(path):
        words = FileOpen(fileName, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in vocab:
                column = vocab.index(key)
                matrix[row][column] = temp[key]
        if (classifier == "ham"):
            FinalList[row] = 0
        elif (classifier == "spam"):
            FinalList[row] = 1
        row += 1
    return matrix, row, FinalList


row_train = 0
row_test = 0

#train matrix including spam and ham
matrix_train, row_train, train_TargetList = createMatrix(matrix_train, train_ham_path, list_Train, row_train, "ham", train_TargetList)
matrix_train, row_train, train_TargetList = createMatrix(matrix_train, train_spam_path, list_Train, row_train, "spam", train_TargetList)

#train matrix including spam and ham
matrix_test, row_test, test_TargetList = createMatrix(matrix_test, test_ham_path, list_Test, row_test, "ham", test_TargetList)
matrix_test, row_test, test_TargetList = createMatrix(matrix_test, test_spam_path, list_Test, row_test, "spam", test_TargetList)


# for each column
def sigmoid(x):
    s_func = 1 / (1 + np.exp(-x))
    return s_func

# Calculate for each file
# Logistic Function (using the formula)
def logisticFunction(totalFiles, totalFeatures, featureMatrix):
    global sigmoidList
    
    for files in range(totalFiles):
        summation = 1.0
        for features in range(totalFeatures):
            summation += featureMatrix[files][features] * weightOfFeature[features]
            
        sigmoidList[files] = sigmoid(summation)

# Updating the weights for convergence
def getWeightUpdate(totalFiles, numOfFeatures, featureMatrix, FinalList):
    global sigmoidList

    for feature in range(numOfFeatures):
        weight = bias
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            y = FinalList[files]
            sigmoidValue = sigmoidList[files]
            weight += frequency * (y - sigmoidValue)

        oldW = weightOfFeature[feature]
        learnedW = (weight * learningRate)
        newW = (learningRate * Lambda * oldW)
        weightOfFeature[feature] += np.subtract(learnedW, newW)
    return weightOfFeature

# Training data
def trainingFunction(totalFiles, numOfFeatures, trainFeatureMatrix, FinalList):
    logisticFunction(totalFiles, numOfFeatures, trainFeatureMatrix)
    getWeightUpdate(totalFiles, numOfFeatures, trainFeatureMatrix, FinalList)

# Classifying data
def classifyData():
    correctSpam = 0
    incorrectSpam = 0
    correctHam = 0
    incorrectHam = 0
    idx = 0
    for file in range(total_TestFiles):
        print('TestFile : '+str(idx+1))
        summation = 1.0
        for i in range(len(list_Test)):
            word = list_Test[i]

            if word in list_Train:
                index = list_Train.index(word)
                weight = weightOfFeature[index]
                wordcount = matrix_test[file][i]

                summation += weight * wordcount

        sigSum = sigmoid(summation)
        if (test_TargetList[file] == 0):
            if sigSum < 0.5:
                correctHam += 1
            else:
                incorrectHam += 1
        else:
            if sigSum >= 0.5:
                correctSpam += 1
            else:
                incorrectSpam += 1
        idx += 1
    print("Lambda:",Lambda)
    print("Number of Iterations:", Iterations)
    print("Accuracy on Spam:",correctSpam,"/",(correctSpam + incorrectSpam),"=",
          round((correctSpam / (correctSpam + incorrectSpam)) * 100,2))
    print("Accuracy on Ham:",correctHam,"/",(correctHam + incorrectHam),"=",
          round((correctHam / (correctHam + incorrectHam)) * 100,2))

    correctlyClassified = correctHam+correctSpam
    totalClassified = correctHam + incorrectHam+correctSpam + incorrectSpam
    print("Overall Accuracy:",correctlyClassified,"/",totalClassified,"=",
          round((correctlyClassified / totalClassified) * 100,2))


print("\nTraining the algorithm - ")
for i in range(int(Iterations)):
    print(i, end=' ')
    trainingFunction(total_TrainFiles, len(list_Train), matrix_train, train_TargetList)

print("Training completed successfully")
print("\nPlease wait for a few minutes while the data is being classified..\n")
classifyData()


