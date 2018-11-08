import numpy as np
import math

def rotation_manipulation(thetas,rotation_angle,old_value):
	"""
		The coefficients are to impose harder constraints on the 
		rotation of the canonical bins. This ensures that they 
		aren't rotated by more than the actual anlge of rotation
		in sum_delta. Bins 0,44,45 and 89 are the canonical bins.
	"""
	sum_thetas = 0
	for i in range(90):
		try:
			sum_thetas+=(thetas(i)-thetas(i+1))**2
		except:
			continue
	sum_delta = 0
	sum_delta = 1000*((thetas[0] - rotation_angle)**2 + (thetas[44]-rotation_angle)**2 + (thetas[45]-rotation_angle)**2 + (thetas[89] - rotation_angle)**2)
	energy_rotation = sum_delta + sum_thetas
	return energy_rotation+old_value

def line_preservation(linethetas, V, uk, Pk, lines, bins,old_value):
	"""
		This function is to ensure that we build a relation between
		the lines and the meshes. This is a function of both V and 
		Theta. 
	"""
	ek = Pk.dot(V)
	for k in range(len(lines)):
	
		U_k = uk[k].dot(np.linalg.inv(np.transpose(uk[k]).dot(uk[k]))).dot(np.transpose(uk[k]))
		theta = linetheta[k]*math.pi/180 
		"""
			Convering the Angle into radians
			List of all the lines with their angles of orientation
		"""
		R_k = np.asarray([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.sin(theta)]])

		val = (R_k.dot(U_k).dot(np.transpose(R_k)) - np.eye(2)).dot(ek[k])
		return linalg.norm(val)

def boundary_preservation(V,old_value):
	"""
		Essentially an energy function designed to impose very strong 
		conditions on the boundary. The penalty is enormous in this 
		minimization problem, for we should try to change the value. 
	"""

def shape_preservation(quad_count,x,y,V,Q,old_value):
	"""
		Another Energy Function mainly aimed towards preserving 
		the shape of the image. There's no penalty in terms of a
		coefficient for this case though, unlike other cases.
	"""
	Vq = Q.dot(V)
	N = quad_count
	for q in range(N):
		Aq = np.asarray([[x[i],-y[i],1,0],
						[y[i],x[i],0,1],
						[x[i+1],-y[i+1],1,0],
						[y[i+1],x[i+1],0,1],
						[x[i+2],-y[i+2],1,0],
						[y[i+2],x[i+2],0,1],
						[x[i+3],-y[i+3],1,0],
						[y[i+3],x[i+3],0,1]])
		S = (Aq.dot(np.linalg.inv(np.transpose(Aq).dot(Aq))).dot(Aq)-eye(8)).dot(Vq)
		val = np.transpose(S).dot(np.transpose(S))


def total_energy(quad_count,x,y,linethetas,Q,V):
	""" 
	This is the total energy function. This is a linear combination
	of all the above components written above. The penalities are 
	reflected in the form of lambdas.
	"""

	lambda_b = 10000000
	lambda_l = 100
	lambda_r = 100
	total_energy = shape_preservation(quad_count,x,y,v,Q) + lambda_b*boundary_preservation() +lambda_l*line_preservation(linethetas, V, uk, Pk, lines, bins) + lambda_r*rotation_manipulation(thetas,rotation_angle)
	
	return total_energy
