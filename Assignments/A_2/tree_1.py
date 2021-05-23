#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:49:02 2021

@author: raa
"""

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import math
import graphviz
import copy
from random import randrange

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    uniq_vec = dict()
    u, indices = np.unique(x, return_inverse=True)
        
    for j in u:
        res = [i for i, val in enumerate(x) if val==j] 
        uniq_vec[j]=res
#    print(uniq_vec)
    return(uniq_vec)




def entropy(y,w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    if w is None:
        w=np.ones((len(y),1),dtype=int)
    
    total=np.sum(w)
    uniq_vec=partition(y)
    
    ent = 0
    
    for i in uniq_vec.keys():
        split_w=[]

        for x in uniq_vec[i]:
            split_w.append(w[x])
        e_y=(np.sum(split_w)/total)
        ent += - e_y * math.log(e_y, 2)
    
#    print(ent)
    return ent


def mutual_information(x, y,w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    if w is None:
        w=np.ones((len(y),1),dtype=int)
        
    e_y=entropy(y,w)
    
#    IG=dict()
    e_split_y=0
    uniq_vec = dict()

    uniq_vec=partition(x)
    n=len(x)
    total_weight=np.sum(w)
    for i in uniq_vec.keys():
        split_y=[]
        split_w=[]
        for x in uniq_vec[i]:
            split_y.append(y[x])
            split_w.append(w[x])
#        print(np.shape(split_y))
        e_split_y += np.sum(split_w) * entropy(split_y,split_w)

    MI = e_y - (e_split_y/total_weight)
#    print(MI)
    return(MI)


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    
    if w is None:
        w=np.ones((len(y),1),dtype=int)
    
    tree={}    
    numcols = len(x[0])
    IG=dict()
    uniq_vec = dict()
    res=0
    attr_val=[]
    attr_val_copy=[]
    tree={}
    
    #Check if the labels of the node are pure
    if len(np.unique(y)) == 1:
        return (y[0])
    
    #Create attribute value pair for the first iteration
    if attribute_value_pairs == None:
        for feature in range(0,numcols):
            for val in np.unique(x[:,feature]):
                attr_val.append((feature,val))
        
    else:
        attr_val=attribute_value_pairs
        
    #Check if all the attribute-value pairs are processed or the max depth is reached
    #Stop creating tree if any of the above conditions meet        
    y_val, y_count = np.unique(y, return_counts=True)   
         
    if len(attr_val) == 0 or depth == max_depth:
        return(y_val[np.argmax(y_count)])
    
    #Determine the best attribute-value pair based on highest mutual information
    for i,v in (attr_val):
        bin_x=(np.array(x)[:, i] == v).astype(int)
        IG[(i,v)]=mutual_information(bin_x,y,w)
    
    best_attr,best_val=max(IG, key=IG.get)

    #Partition the dataset based on best attribute-value pair determined
    partitions = partition((np.array(x)[:, best_attr] == best_val).astype(int))

    #Creates a copy of attribute-value pairs and removes the best attribute-value pair for next split
    attr_val_copy = attr_val.copy()
    attr_val_copy.pop(attr_val.index((best_attr,best_val)))

    #For each of the partitions created partitionwd x, y and recursively call id3
    depth = depth + 1
#    for val, indices in partitions.items():
#        split_y=[]
#        split_x=[]
#        for k in indices:
#            split_y.append(y[k])
#            split_x.append(np.array(x)[k,:])
#            decision=bool(val) 
    
    for split_val, indices in partitions.items():
        split_x = x.take(indices, axis=0)
        split_y = y.take(indices, axis=0)
        decision = bool(split_val)
           
    
        tree[(best_attr, best_val, decision)] = id3(split_x, split_y, attribute_value_pairs=attr_val,
                                            max_depth=max_depth, depth=depth, w=w)
            
    return tree   



def predict(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    
    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_trees) is dict:
                label = predict(x, sub_trees)
            else:
                label = sub_trees

            return label


def compute_error(y_true, y_pred, w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    
    if w is None:
        w=np.ones((len(y_true),1),dtype=int)
        
    n=len(y_true)
    err = [ w[i] * (y_true[i] != y_pred[i]) for i in range(n)]
    return (np.sum(err) / np.sum(w))


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


#Method to create sampling with replacement
def subsample(x,y):
    
    index_values=[]
	
    n_sample = len(x)
    while len(index_values) < n_sample:
        index = randrange(len(x))
        index_values.append(index)
        
    x_sample=(x[index_values])
    y_sample=(y[index_values])

    return x_sample,y_sample

def bagging(x, y, max_depth, num_trees):
    h_ens={}
    alpha_i=1
    
    #Create ensemble set of trees by sampling from dataset
    for i in range (0,num_trees):
        x_sample,y_sample=subsample(x,y)
        decision_tree=id3(x_sample, y_sample, max_depth=max_depth)
        h_ens[i]=(alpha_i,decision_tree)
    
    return h_ens


def predict_example(x, h_ens):
    
    y_pred=[]
    
    for x_row in x:
        pred=[]
        n=0
        for bag in h_ens.keys():
            alpha,tree=h_ens[bag]
            p=predict(x_row, tree)
            pred.append(alpha*p)        
            n=n+alpha #total weight

        val=np.sum(pred)/n
        if val >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return y_pred


def boosting(x, y, max_depth,num_stumps):
    rows, cols = np.shape(x)
    d = (1/rows)
    weight = np.full((rows, 1), d)
    h_ens = {}
    alpha_i = 0
    trn_pred = []

    for stump in range(0, num_stumps):

        tree = id3(x, y, max_depth=max_depth, w=weight)
        
        #Predict using the decision stump and compute error
        trn_pred = [predict(x_row, tree) for x_row in x]
        err = compute_error(y, trn_pred, weight)
        alpha_i = 0.5 * np.log((1-err)/err)
        
        #Recompute weight based on error
        pre_weight = weight
        weight = []
        for i in range(rows):
            if y[i] == trn_pred[i]:
                weight.append(pre_weight[i] * np.exp(-1*alpha_i))
            else:
                weight.append(pre_weight[i] * np.exp(alpha_i))
        d_total = np.sum(weight)
        weight = weight / d_total
        
        h_ens[stump] = (alpha_i, tree)
    return h_ens
    
    
def compute_confusion_matrix(true, pred):

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result.astype(int)

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
