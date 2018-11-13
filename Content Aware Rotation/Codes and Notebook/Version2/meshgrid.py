import matplotlib.pyplot as plt
import pylab
"""
	This file is used to draw the grid. We will initially draw 
	the grid consisting of about 400 square quads. Then later on
	we will show the warped mesh, following which, we will show the 
	actually warped image.
"""

def drawmesh(image):
	"""
		This is for drawing the mesh.
	"""

def warpmesh(image,X,Y):
	final = np.zeros(image.shape,Vx,Vy)
	"""
		This is to warp the image. We shall use a barycentric coordinate
		based approach to perform the same. 
	"""
	lineX = np.linspace(1,1,X)
	lineY = np.linspace(1,1,Y)
	numX = len(lineX)
	numY = len(lineY)
	gridX = np.ones(numY).dot(lineX)
	gridY = lineY.dot(np.ones(numX))
	sampleX = gridX.transpose().reshape((numX*numY))
	sampleY = gridY.transpose().reshape((numX*numY))
	sampleXY = np.zeros((2,sampleX.shape[1]))
	ori_sampleXY = np.zeros(sampleXY.shape)
	tmpI = np.zeros((3,sampleXY.shape[1]))
	count = np.zeros((3,samplyXY.shape[2]))

	for i in range(cx):
		for j in range(cy):
			v4 = np.array([i*(cy+1)+j+1,i*(cy+1)+j+2,(i+1)*(cy+1)+j+1,(i+1)*(cy+1)+j])
			for k in range(2):
				if k ==0:
					tri_v = np.array([v4[0],v4[1],v4[3]])
				else:
					tri_v = np.array([v4[0],v4[2],v4[3]])
				cornerX = Vx[tri_v]
				cornerY = Vy[tri_v]
				cornerXY = np.append(cornerX, cornerY, axis = 0)
				d_sampleXY = sampleXY - cornerXY[:,2].dot(np.ones((sampleXY,2)))
				T = cornerXY[:,0:2] - cornerXY[:,2].dot(np.ones((1,2)))
				lamb = np.linalg.inv(T).dot(d_sampleXY)
				index_table = (lamb[1,:]<=1)