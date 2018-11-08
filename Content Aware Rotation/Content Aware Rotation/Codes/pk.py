import numpy as np
from math import floor, ceil
"""
	This is a module to compute Pk; the breakdown of the eK's in the paper
	into Pk times V. 
"""

"""
	x,y are the same as vertexX and vertexY => len(x) =cx and len(y) = cy;
	N is the number of vertices
	dx, dy are the increments that we use across the grids.
"""
def computepk(lines,dx,dy,N,x,y):

	Pk = np.zeros(len(lines),N)
	# PkV = np.zeros(len(lines),N)
	# Pk1 = np.zeros(len(lines),N)
	# Pk2 = np.zeros(len(lines),N)

	for i in range(len(lines)):

		x1 = lines[i][0]
		y1 = lines[i][1]
		x2 = lines[i][2]
		y2 = lines[i][3]

		min_x = min(x1,x2) 
		min_y = min(y1,y2)
		"""
			This is for extracting the corresponding Quad to be
			found for Bilinear Interpolation
		"""
		xleft = floor(xmin/dx)+1
		xright= xleft+1

		ytop = floor(ymin/dy)+1
		ybottom= ytop+1

		"""
			Initialising the matrices for bilinear interpolation.
			We'll subtracting these matrices and then constructing the needed matrix
		"""

		Pkmatrix1 = np.zeros(len(y)+1,len(x)+1)
		Pkmatrix2 = np.zeros(len(y)+1,len(x)+1)
		PkmatrixV = np.zeros(len(y)+1,len(x)+1)

		Pkmatrix1[ytop,xleft] = ((x[xright]-x1)/dx)*((y[ybottom]-y1)/dy)
		Pkmatrix1[ytop,xright]= ((x1-x[xleft])/dx)*((y[ybottom]-y1)/dy)
		Pkmatrix1[ybottom,xleft]= ((x[xright]-x1)/dx)*((y1-y[ytop])/dy)
		Pkmatrix1[ybottom,xright]=((x1-x[xleft])/dx)*((y1-y[ytop])/dy)

		Pkmatrix2[ytop,xleft] = ((x[xright]-x2)/dx)*((y[ybottom]-y2)/dy)
		Pkmatrix2[ytop,xright]= ((x2-x[xleft])/dx)*((y[ybottom]-y2)/dy)
		Pkmatrix2[ybottom,xleft]= ((x[xright]-x2)/dx)*((y2-y[ytop])/dy)
		Pkmatrix2[ybottom,xright]=((x2-x[xleft])/dx)*((y2-y[ytop])/dy)

		PkmatrixV[ytop,xleft] = 1
		PkmatrixV[ytop,xright]= 1
		PkmatrixV[ybottom,xleft]= 1
		PkmatrixV[ybottom,xright]=1

		Pk[i] = np.reshape(Pkmatrix2- Pkmatrix1,(1,N))
		# PkV[i] = np.reshape(PkmatrixV,(1,N))
		# Pk1[i] = np.reshape(Pkmatrix1,(1,N))
		# Pk2[i] = np.reshape(Pkmatrix2,(1,N))

	return Pk



		

		
