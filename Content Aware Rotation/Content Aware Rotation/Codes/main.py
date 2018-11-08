""" 

Content Aware Rotation, In ICCV 2013	
Authors: Kaiming He, Huiwen Chan, Jian Sun

An implementation by S Deepak Narayanan, Indian Institute of Technology Gandhinagar

"""


""" 
Standard Imports for the rest of the program 
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np 
import random as rd 
from energyfunction import total_energy
from optimization import *
from line_matrix import *
from shape_matrix import *
from extractlines import quantize_and_get
from boundary_matrix import *
"""
Here I amn initialising all the parameters
"""

rotation_angle = np.inf
img = cv.imread('fig1.png')


Y,X = img.shape[:2]
cx= int(X/30)
cy = int(Y/30)

n_quads = cx*cy
number_of_vertices = (cx+1)*(cy+1)
x_len = (X-1)/cx
y_len = (Y-1)/cy


threshold = 16
linesegthreshold = (x_len**2 + y_len**2)/64

temp = 0
vertexX = np.zeros(1)
while(1):
    temp+=x_len
    temp = (round(temp,10))
    vertexX = np.append(vertexX,temp)
    if temp>X-2:
        break
temp = 0
vertexY = np.zeros(1)
while(1):
    temp+=y_len
    temp = (round(temp,10))
    vertexY = np.append(vertexY,temp)
    if temp>Y-2:
        break
gridX, gridY = np.meshgrid(vertexX,vertexY)
Vx = np.reshape(gridX,number_of_vertices,1)
Vy = np.reshape(gridY,number_of_vertices,1)
V = np.zeros((number_of_vertices*2))
for i in range(number_of_vertices):
    V[2*i] = Vx[i]
    V[2*i+1] = Vy[i]

print(X,Y)
delta = 6.1

print("Line Extraction and Quantization Begin....")
lines = quantize_and_get(X,Y,threshold,linesegthreshold,x_len,y_len,delta)
print("Line Extraction and Quantization Done .... ")
print("Forming the functions for Shape, Line, Boundary and Rotation Constraints...")

#print(gridX)

"""
    Intialising Thetas Now
"""

# thetas = np.ones((90,1))*delta

# line = formline(lines,Pk,lambda_l,thetas,number_of_vertices,dx,dy,x,y)
# shape = formshape(x,y,number_of_vertices,quad_count)
# boundary,b = formboundary()
# for iter_no in range(1,11):
#     print('Currently we are in iteration number ',iter_no,' of optimization')
#     V = fix_theta_solve_V(line,shape,boundary,b)
#     print('First Half of iteration ',iter_no,' done....')
#     thetas = fix_V_solve_theta()


val1,val2  = (formboundary(number_of_vertices,X,Y,gridX,gridY))