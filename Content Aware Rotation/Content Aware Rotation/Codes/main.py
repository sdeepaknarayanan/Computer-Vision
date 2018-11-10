""" 

Content Aware Rotation, In ICCV 2013	
Authors: Kaiming He, Huiwen Chan, Jian Sun

An implementation by S Deepak Narayanan, Indian Institute of Technology Gandhinagar

"""


""" 
Standard Imports for the rest of the program 
"""
import pandas as pd
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
from pk import *

"""
Here I amn initialising all the parameters
"""

rotation_angle = np.inf
img = cv.imread('fig1.png')


Y,X = img.shape[:2]

"""
	Total Number of Quads created = 900. This is very 
	deterimental in deciding the amount of time the 
	program takes to run
"""

"""
	cx = Number of Quads along the horizontal direction
	cy = Number of Quads along the vertical direction
	n_quads = Total number of quads
	number_of_vertices  = Total Number of Mesh Vertices
	x_len = Distance between subsequent Quads along x_direction
	y_len = Distance between subsequent Quads along y_direction
	threshold and linesegthreshold = Used for choosing the correct 
		Line Segements among those that were detected.
	vertexX = List of all the X_coordinates of the grid points
	vertexY = List of all the Y_coordinates of the grid points
	gridX,gridY =  Meshgrid that we form using these vertices
"""

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

thetas = np.ones((90,1))*delta

Pk_all,line = formline(lines,number_of_vertices,x_len,y_len,vertexX,vertexY)
shape_preservation = formshape(vertexX,vertexY,number_of_vertices,n_quads)
boundary,b = formboundary(number_of_vertices,X,Y,gridX,gridY)

lambda_l = 100
lambda_b = 100000000
lambda_r = 100  

"""
	Optimization Begin
"""
print('Boundary Size is ',boundary.shape)
print('Shape Size is ',shape_preservation.shape)
print('Line Size is ',line.shape)
print('PK_all is ',Pk_all.shape)
print('b is ',b.shape)
n = number_of_vertices
k = len(lines)
print(n,k)
sdelta = np.zeros(len(thetas))
for number_of_iteration in range(1,11):
	V_new = fix_theta_solve_v(line,shape_preservation,boundary,b,lambda_l,lambda_r,lambda_b,n,k)
	df = pd.DataFrame(V_new)
	df.to_csv('V1.csv')
	print(np.linalg.norm(V-V_new)**2)
	break