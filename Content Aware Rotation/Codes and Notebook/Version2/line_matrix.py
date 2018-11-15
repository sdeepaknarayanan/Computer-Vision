import numpy as np
import math
from shape_matrix import *
from extractlines import quantize_and_get
from boundary_matrix import *
from pk import *

"""
    Two Functions are defined here. 
    getuk() and formline()
    getuk - Used to get the list of all the unit vectors of the line that we're
    using under consideration.
    formline() - This is used in the formation of the Line Matrix that we'd be optimising
    on. Here, we need to compute Pk, that is not explicitly given in the paper.
"""

def getuk(lines):
    uk = np.zeros((len(lines),2))
    for i in range(len(lines)):
        x =  lines[i][2] - lines[i][0]
        y = lines[i][3] - lines[i][1]
        temp = np.array([x,y])
        try:
            temp = temp/np.linalg.norm(temp)
            uk[i] = temp
        except:
            temp = np.array([0,0])
            uk[i] = temp
    return uk


def formline(lines,number_of_vertices,dx,dy,x,y,thetas):
    U_final = np.zeros((len(lines),2,2))
    """
        This is to form the Energy Function for 
        adding the constraint to preserve the line segments.
        This is a very important part of the project.
        Standard Initialisations below.
    """
    uk = getuk(lines)
    uk = np.matrix(uk)
    N = number_of_vertices
    Pk = computepk(lines,dx,dy,N,x,y)
    matrix = np.zeros((2*N,2*N))
    tempPk1 = np.zeros((2*N))
    tempPk2 = np.zeros((2*N))
    Pk_ = np.zeros(((2*len(lines)),2*N))
    
    for i in range(len(lines)):
        Pk_final = np.zeros((2,2*N))
        for j in range(N):
            tempPk1[2*j] = Pk[i][j]
            tempPk2[2*j+1] = Pk[i][j]
        Pk_final[0] = tempPk1
        Pk_final[1] = tempPk2
        theta = thetas[int(lines[i][5]) - 1]*np.pi/180
        U = uk[i].transpose().dot(uk[i])
        val = (np.array(uk[i]).dot(np.transpose(np.array(uk[i]))))
        U = np.array(U)*val
        U_final[i] = U
        Rk = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        temp = (Rk.dot(U).dot(np.transpose(Rk)) - np.eye(2)).dot(Pk_final)
        inter = np.transpose(temp).dot(temp)
        matrix+=inter
        Pk_[2*(i)] = Pk_final[0]
        Pk_[2*i+1] = Pk_final[1]
    """
        We need Uk in the optimisation of thetas, 
        also, we need to keep changing Pk as it is an optimisation
        of theta and V. Important for each iteration.
    """
    return Pk_,matrix,U_final