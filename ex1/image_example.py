#!/usr/bin/env python
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

# Before using this you have to install the Python Pillow and numpy libraries.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from PIL import Image

def sobel_edge_detection(img):
    color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_axis = cv2.Sobel(color,cv2.CV_64F,1,0)
    Y_axis = cv2.Sobel(color,cv2.CV_64F,0,1)
    x = np.power(X_axis,2)
    y = np.power(Y_axis,2)
    return np.sqrt(x+y)

def minimum_energy(edge_img):
    min_energy = np.zeros(edge_img.shape)
    N = edge_img.shape[0]
    M = edge_img.shape[1]

    for j in range(0,M):
        min_energy[N-1][j] = edge_img[N-1][j]

    for i in range(N-2,-1,-1):
        for j in range(0,M):
            left = max(0,j-1)
            right = min(M-1,j+1)
            min_energy[i][j] = edge_img[i][j] + min(min_energy[i+1][left:right+1])

    return min_energy

def delete_seam(img,min_energy):
    N = min_energy.shape[0]
    M = min_energy.shape[1]
    
    index = 0
    value = min_energy[0][0]

    for i in range(1,N):
        if(min_energy[0][i] < value):
            index = i
            value = min_energy[0][i]

    new_img = np.zeros((img.shape[0],img.shape[1]-1,img.shape[2]),dtype=np.uint8)

    new_img[0] = np.concatenate((img[0][0:index],img[0][index+1:img.shape[1]]))
    for i in range(1,N):
        center = index
        min = min_energy[i][center]
        if(center>0 and min_energy[i][center-1] < min):
            min = min_energy[i][center-1]
            index = center -1

        if(center<M-1 and min_energy[i][center+1] < min):
            min = min_energy[i][center+1]
            index = center+1
        new_img[i] = np.concatenate((img[i][0:index],img[i][index+1:img.shape[1]]))
    return new_img

def number_of_seams(img,n_seam):
    new_img = img
    for i in range(0,n_seam):
        min_energy = minimum_energy(sobel_edge_detection(new_img))
        new_img = delete_seam(new_img,min_energy)
    return new_img

img = imageio.v2.imread('tower.jpg')
obtain_shape = (100,100,3)
n_seams = np.subtract(img.shape,obtain_shape)
new_img = number_of_seams(img,int(n_seams[np.nonzero(n_seams)]))
new_img = Image.fromarray(new_img)
new_img.save('out.png')