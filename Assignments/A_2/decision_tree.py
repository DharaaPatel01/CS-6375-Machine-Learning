# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
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


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    unique_Dict = dict()
    unique_X = np.unique(x)
    
    for val in unique_X:
        arr = [j for j, value in enumerate(x) if value==val] 
        unique_Dict[val] = arr
    return unique_Dict


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    p_y = 0.0
    entr = 0.0
    unique_Dict = partition(y)
    
    for key in unique_Dict:        
        arrTemp = []
        for i in range(len(y)):
            arrTemp.append(0)
        for u in unique_Dict[key]:
            arrTemp[u]=1
        
        p_y = float(sum(arrTemp))/float(len(y))
        entr += (- p_y * math.log(p_y,2))
    return entr

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    
    entr_y = entropy(y)
    entr_yx = 0.0
    p_x = float(sum(x))/float(len(x))
    
    # i=0
    unique_y = partition(y)
    for val in unique_y:
        idx = []
        for i in range(len(y)):
            idx.append(0)
        for u in unique_y[val]:
            idx[u]=1
            
        p_y = 0.0
        for i in range(len(y)):
            if(idx[i] and x[i]):
                p_y += 1.0
        print("x >> ",x)
        p_yx = float(p_y)/float(sum(x))
        if p_yx!=0.0:
            entr_yx +=(- p_yx * math.log(p_yx,2))
    
    return entr_y-(p_x * entr_yx)


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
        print("bin >> ", bin_x)
        IG[(i,v)]=mutual_information(bin_x,y)
    
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
"""
    tree={}

    if len(np.unique(y))==1 and (y[0] == 0 or y[0] == 1):
        return y[0]
    
    if len(x)==0 or len(y)==0:
        return None


    if attribute_value_pairs!=None and len(attribute_value_pairs)==0:
        #print "here"
        ydash=set(y)
        maxcount=0
        maxy=0
        for y1 in ydash:
            count=0
            for i in range(len(y)):
                if y[i]==y1:
                    count+=1
            if count>maxcount:
                maxcount=count
                maxy=y1
        return maxy

    if depth==max_depth:
        #print "here too"
        ydash = set(y)
        maxcount = 0
        maxy=0
        for y1 in ydash:
            count = 0
            for i in range(len(y)):
                if y[i] == y1:
                    count += 1
            if count > maxcount:
                maxcount = count
                maxy=y1
        return maxy


    avmi={}
    maxmi=0
    maxcol=0
    maxattr=0
    ab=np.shape(x)
    #print (np.shape(x)[1])
    # xtemp=x.tolist()

    for i in range(ab[1]):
        d = partition([row[i] for row in x])

        for key in d:
            #if (i,key) not in avpairs:
                indices=d[key]
                
                arrTemp = []
                for i in range(len(x)):
                    arrTemp.append(0)
                for idx in indices:
                    arrTemp[idx] = 1
                mi=mutual_information(arrTemp,y)
                #print(temp)
                avmi[(i,key)]=mi
    if attribute_value_pairs==None:
        attribute_value_pairs=list(avmi.keys())
    else:
        for key in avmi.keys():
            if key not in attribute_value_pairs:
                avmi.pop(key)

    if len(avmi)==0:
        return

    print(avmi)
    print("attr >> ", attribute_value_pairs)
        
    maxmi = max(avmi.values())
    #print(maxmi)
    
    key;
    for k in avmi:
        if avmi[k] == maxmi:
            key = k    
        
    (col, attr) = key
    print(col,attr)
    print(key)
    #avmi.pop(key)
    
    
    best_col_x = [row[col] for row in x] 
    true_y=[]
    false_y=[]
    x_true=[]
    x_false=[]
    for i in range(len(y)):
        if best_col_x[i]==attr:
            true_y.append(y[i])
            x_true.append(x[i])
        else:
            false_y.append(y[i])
            x_false.append(x[i])


    cloned=attribute_value_pairs[:]
    cloned.remove(key)

    tree[(col,attr,True)]=id3(x_true,true_y,cloned,depth+1,max_depth)

    tree[(col, attr, False)]=id3(x_false,false_y,cloned,depth+1,max_depth)
#    print tree

    return tree
"""

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


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
