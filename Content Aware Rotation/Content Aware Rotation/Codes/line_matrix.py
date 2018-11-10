import numpy as np
import math
from pk import *
import pandas as pd

"""
	Incomplete as on 10th November! Bad - This is horrible!
"""

def getuk(lines):
	uk = np.zeros((len(lines),2))
	for i in range(1,len(lines)):
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


def formline(lines,number_of_vertices,dx,dy,x,y):
	"""
		This is to form the Energy Function for 
		adding the constraint to preserve the line segments.
		This is a very important part of the project.
	"""
	uk = getuk(lines)
	uk = np.matrix(uk)
	df = pd.DataFrame(uk)
	df.to_csv('UK.csv')
	N = number_of_vertices
	Pk = computepk(lines,dx,dy,N,x,y)
	matrix = np.zeros((2*N,2*N))
	tempPk1 = np.zeros((2*N))
	tempPk2 = np.zeros((2*N))
	Pk_ = np.zeros(((2*len(lines)),2*N))
	
	for i in range(1,len(lines)):
		Pk_final = np.zeros((2,2*N))
		for j in range(N):
			tempPk1[2*j] = Pk[i][j]
			tempPk2[2*j+1] = Pk[i][j]
		df = pd.DataFrame(tempPk1)
		df.to_csv('temppk1.csv')
		#break
		Pk_final[0] = tempPk1
		Pk_final[1] = tempPk2
		df = pd.DataFrame(Pk_final)
		df.to_csv('Pkf.csv')
		theta = lines[i][5]*np.pi/180
		U = uk[i].transpose().dot(uk[i])
		val = (np.array(uk[i]).dot(np.transpose(np.array(uk[i]))))
		U = np.array(U)*val
		Rk = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
		
		temp = (Rk.dot(U).dot(np.transpose(Rk)) - np.eye(2)).dot(Pk_final)
		inter = np.transpose(temp).dot(temp)
		matrix+=inter
		Pk_[2*(i-1)] = Pk_final[0]
		Pk_[2*i-1] = Pk_final[1]
	
	Pk_final = np.zeros((2,2*N))
	for j in range(N):
		tempPk1[2*j] = Pk[len(lines)-1][j]
		tempPk2[2*j+1] = Pk[len(lines)-1][j]
	Pk_[2*(len(lines)-1)] = Pk_final[0]
	Pk_[2*len(lines)-1] = Pk_final[1]
	df = pd.DataFrame(Pk_)
	df.to_csv('AllDone.csv')
	return Pk_,matrix

