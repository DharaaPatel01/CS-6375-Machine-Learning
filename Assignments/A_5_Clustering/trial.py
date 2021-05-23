#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:14:13 2021

@author: raa
"""

from matplotlib import pyplot as io 
import numpy as np
from PIL import Image

# Image to array
img1 = io.imread('Koala.jpg') #image is saved as rows * columns * 3 array print (img1)

#Array to image file
array = np.zeros([10,20,3], dtype = np.uint8)
array[:,:10] = [255, 128, 0]    # Orange left side
array[:,10:]  = [0,0,255]   # Blue right side
print(array)
    
img2 = Image.fromarray(array)
img2.save('testrg.png')