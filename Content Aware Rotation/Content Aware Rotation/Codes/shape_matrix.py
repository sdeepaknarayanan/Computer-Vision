import numpy as np


"""
	This function computes the value of StS, see report for further details
"""

def formshape(x,y,number_of_vertices,quad_count): # X and Y are VertexX and Vertexy
	N = number_of_vertices
	matrix = np.zeros((N,N))
	for i in range(len(y)):
		for j in range(len(x)):
			""" Each Quad is taken care of. (i,j) = Refers to the 
				top left vertex of every quad
			"""
			try:
				Aq = np.asarray([[x[j],-y[i],1,0],
								[y[i],x[j],0,1],
								[x[j],-y[i+1],1,0],
								[y[i+1],x[j],0,1],
								[x[j+1],-y[i],1,0],
								[y[i],x[j+1],0,1],
								[x[j+1],-y[i+1],1,0],
								[y[i+1],x[j+1],0,1]])
				# Decompose Vq as Q times V. 
				# Dont care about V for the time being
				# |x| = |y| = Number of Vertices making up the grid.
				# Non Trivial Decomposition
				Q = np.zeros(8,2*N)
				Q[0][2*((j-1)*(len(y)+1)+i)-2] = 1
				Q[1][2*((j-1)*(len(y)+1)+i)-1] = 1
				Q[2][2*((j-1)*(len(y)+1)+i)] = 1
				Q[3][2*((j-1)*(len(y)+1)+i)+1] = 1
				Q[4][2*((j)*(len(y)+1)+i)-2] = 1
				Q[5][2*((j)*(len(y)+1)+i)-1] = 1
				Q[6][2*((j)*(len(y)+1)+i)] = 1
				Q[7][2*((j)*(len(y)+1)+i)] = 1

				temp_matrix = (Aq.dot(np.linalg.inv(Aq.transpose().dot(Aq))).dot(Aq.transpose()) - np.eye(8)).dot(Q)
				matrix+=temp_matrix.transpose().dot(temp_matrix)
			except:
				continue
				"""
					Exception Handling at the corner of the Grid
				"""
	matrix = matrix/quad_count
	return matrix





