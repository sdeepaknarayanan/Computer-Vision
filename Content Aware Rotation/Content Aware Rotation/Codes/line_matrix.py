import numpy as np
import math
from pk import computepk


def getuk(lines):
	uk = np.zeros((len(lines),2))
	for i in range(len(lines)):
		x =  lines[i][2] - lines[i][0]
		y = lines[i][3] - lines[i][1]
		temp = np.array([x,y])
		temp = temp/np.linalg.norm(temp)
		uk[i] = temp
	return uk

def formline(lines,Pk,lambda_l,thetas,number_of_vertices,dx,dy,x,y):
	"""
		Linestheta has the two tuples consisting of the following.
		[(Line Number, Rotation Angle)] for every line detected.
	"""
	uk = getuk(lines)
	N = number_of_vertices
	Pk = computepk(lines,dx,dy,N,x,y)
	
	matrix = np.zeros((2*N,2*N))
	K = len(linestheta)
	for k in range(len(lines)):
		theta = thetas[linestheta[k][1]]*(math.pi)/180.
		Rk = np.asarray([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
		U_k= uk[k].dot(np.linalg.inv(np.transpose(uk[k]).dot(uk[k]))).dot(np.transpose(uk[k]))
		temp = (Rk.dot(U_k).dot(np.transpose(Rk)) - eye(2)).dot(Pk[k])
		matrix+=temp
	matrix = matrix/K
	return matrix

