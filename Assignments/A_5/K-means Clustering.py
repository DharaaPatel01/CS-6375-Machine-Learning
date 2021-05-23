#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:38:46 2021

@author: raa
"""

import sys
import numpy as np
from skimage import io, img_as_float
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

if (len(sys.argv) != 4):  
    sys.exit("Please give valid Arguments- ")
else:
    inputImg = sys.argv[1]
    K = int(sys.argv[2])
    Iterations = int(sys.argv[3])
    
def k_means_clustering(img_vectors, k, num_of_iterations):
    labels = np.full((img_vectors.shape[0],), -1)
    cluster_proto = np.random.rand(k, 3)
    
    print('Iterating, please wait for a while...')
    for i in range(num_of_iterations):
        #print('Iteration: ' + str(i + 1))
        points_label = [None for k_idx in range(k)]

        for rgb_idx, rgb_val in enumerate(img_vectors):
            rgb_row = np.repeat(rgb_val, k).reshape(3, k).T

            closest_label = np.argmin(np.linalg.norm(rgb_row - cluster_proto, axis=1))
            labels[rgb_idx] = closest_label

            if (points_label[closest_label] is None):
                points_label[closest_label] = []
            points_label[closest_label].append(rgb_val)

        for k_idx in range(k):
            if (points_label[k_idx] is not None):
                new_cluster_prototype = np.asarray(points_label[k_idx]).sum(axis=0) / len(points_label[k_idx])
                cluster_proto[k_idx] = new_cluster_prototype

    return (labels, cluster_proto)

def get_closest_centroids(X,c):
    K = np.size(c,0)
    arr = np.empty((np.size(X,0),1))
    idx = np.zeros((np.size(X,0),1))
    
    for i in range(0,K):
        y = c[i]
        tmp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,tmp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def compute_centroids(X,idx,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

def plot_colors_by_color(name, img_vectors):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb in img_vectors:
        ax.scatter(rgb[0], rgb[1], rgb[2], c=rgb, marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')

def plot_colors_by_label(name, img_vectors, labels, cluster_proto):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb_i, rgb in enumerate(img_vectors):
        ax.scatter(rgb[0], rgb[1], rgb[2], c=cluster_proto[labels[rgb_i]], marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')

def get_compressed_image():
    image = io.imread(inputImg)[:, :, :3] 
    image = img_as_float(image)
    image_dimensions = image.shape
    
    img_vectors = image.reshape(-1, image.shape[-1])
    labels, color_centroids = k_means_clustering(img_vectors, k=K, num_of_iterations=Iterations)
    
    output_image = np.zeros(img_vectors.shape)
    for i in range(output_image.shape[0]):
        output_image[i] = color_centroids[labels[i]]
    
    output_image = output_image.reshape(image_dimensions)
    print('Saving the Compressed Image')
    
    imgName = sys.argv[1].strip('.jpg') + sys.argv[2] +".jpg"
    io.imsave(imgName, output_image)
    print('Image Compression Completed')
    
    info = os.stat(inputImg)
    print("Image size before compression : ",round(info.st_size/1024, 2),"KB")
    
    info = os.stat(imgName)
    print("Compressed Image size : ",round(info.st_size/1024, 2),"KB")

get_compressed_image()


