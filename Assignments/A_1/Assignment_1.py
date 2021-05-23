#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:42:39 2021

@author: raa
"""

import numpy as np                       # For all our math needs
import math                              # For all our math needs  
import matplotlib.pyplot as plt          # For all our plotting needs

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

# 1a
# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    phi = []
    for value in X:
        z = []
        for dim in range(0,d+1):
            z.append(np.power(value, dim))
        phi.append(z)
    phi = np.asarray(phi)
    return phi

# 1b
# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    a = np.dot(np.linalg.inv(np.dot(Phi.transpose(),Phi)),Phi.transpose())
    W = np.dot(a,y)
    return W

# 1c
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    y_Pred = np.dot(Phi,w)
    err = (y_Pred - y) **2
    sum1 = np.sum(err)
    meanSq_Err = sum1 / len(y)
    return meanSq_Err

# 1d
w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

# Discussion:
# Looking at the plot below, the value of d that will generalize the best should be 18, 
# where the both test and validation errors have one of the lowest values as well as
# the margin between the two is less.

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# 2a
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    phi = []
    for x in X:
        z = []
        for b in B:
            z.append(np.exp(-gamma*(np.power(x-b, 2))))
        phi.append(z)
    phi = np.asarray(phi)
    return phi

# 2b
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
    a = np.linalg.inv(np.dot(Phi.transpose(),Phi) + (lam * np.identity(len(Phi))))
    b = np.dot(Phi.transpose(), y)
    W = np.dot(a,b)
    return W

# 2c
w_rad = {}               # Dictionary to store all the trained models
lambda_rad = {}          # Store all lambda values
validationErr_rad = {}   # Validation error of the models
testErr_rad = {}         # Test error of all the models

lam = 0.001
i = 0

Phi_trn_rad = radial_basis_transform(X_trn, X_trn, gamma = 0.1)
Phi_val_rad = radial_basis_transform(X_val, X_trn, gamma = 0.1)
Phi_tst_rad = radial_basis_transform(X_tst, X_trn, gamma = 0.1)

while (lam <= 1000):
    w_rad[i] = train_ridge_model(Phi_trn_rad,y_trn,lam)
    lambda_rad[i] = math.log10(lam)
    validationErr_rad[i] = evaluate_model(Phi_val_rad, y_val, w_rad[i])
    testErr_rad[i] = evaluate_model(Phi_tst_rad, y_tst, w_rad[i])
    i = i + 1
    lam = lam * 10
    
# Plot all the models
plt.figure()
plt.plot(lambda_rad.values(), validationErr_rad.values(), marker='o', linewidth=3, markersize=12)
plt.plot(lambda_rad.values(), testErr_rad.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('Validation/Test Error', fontsize=16)
plt.xticks(list(lambda_rad.values()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)

# 2d
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(0,7):
    x_d_rad = radial_basis_transform(x_true, X_trn, gamma = 0.1)
    y_d_rad = x_d_rad @ w_rad[d]
    plt.plot(x_true, y_d_rad, marker='None', linewidth=2)

plt.legend(['true'] + list(lambda_rad.values()))
plt.axis([-8, 8, -15, 15])

# Discussion:
# According to the plot below, the lowest lambda value forms the most overfitted curve(in comparison). 
# So, we can infer that the lower the lambda the overfitted the curve and thus the linearity of the model decreases 
# with the decrease in the lambda.
