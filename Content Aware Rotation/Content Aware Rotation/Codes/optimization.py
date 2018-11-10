import numpy as np 
import matplotlib.pyplot as plt 
from line_matrix import *
from shape_matrix import *
import math

def fix_theta_solve_v(line,shape,boundary,b,lambda_l,lambda_r,lambda_b,n,k):
	"""
		This is a Quadratic in V and can be solved by using 
		sparse linear equations. Quadratic Optimization problem.
	"""
	print('Solving the sparse linear system of equations now... - Theta is a constant; Solving for V')
	A = shape/n + lambda_b*(np.transpose(boundary).dot(boundary)) + lambda_l*line/k
	b = -lambda_b*(np.transpose(boundary).dot(b))
	V = np.linalg.solve(A,b)
	return V


	
def fix_v_solve_theta(lines,thetas,V,rotation_angle,Pk,dx,dy,N,x,y,sdelta):
	"""
		This is a non trivial function to solve. Use the half
		quadratic method to solve this equation.
	"""
	k = len(lines)
	delta = rotation_angle
	beta_min = 1
	beta_max = 10000
	beta = beta_min
	M = len(thetas)
	A = np.zeros(k,len(thetas))
	for k in range(1,len(lines)):
		temp=  lines[5]
		A[k][temp] = 1
	T_1 = np.zeros((M,M))
	T_1[0:M-1,1:M] = np.eye(M-1)
	T_1[M-1,0] = 1
	D = np.eye(M,M) - T_1
	Pk = computepk(lines,dx,dy,N,x,y)
	Vx = np.zeros(N)
	Vy = np.zeros(N)
	for i in range(N):
		Vx = V[2*i]
		Vy = V[2*i+1]
	e_x = Pk.dot(Vx)
	e_y = Pk.dot(Vy)
	e = np.array([e_x,e_y,np.acos(e_x/math.sqrt(e_x**2 + e_y**2))/pi*180])
	phi_k = np.zeros(k)
	phi = zeros(k,100)

	while(beta<=beta_max):

		"""
			Fix Theta, update phi. This is a single variable equation
			and is solved by constant lookups, though gradient descent 
			can be used.
		"""
		for i in range(len(lines)):
			ek_uk = e[i][2] - lines[i][4]
			bin = lines[i][5]
			theta_k = thetas(bin)


		"""
			Fix phi, update theta. Quadratic in Theta and can be
			solved by using linear system of equations.
		"""
		H = beta*(np.transpose(A).dot(A)) + lambda_r*np.diag(sdelta) + lambda_r*np.transpose(D).dot(D)
		h = beta*np.transpose(A).dot(phi_k) + lambda_r*diag(sdelta)*np.ones((M,1))*delta
		thetas = np.linalg.solve(H,h)
		
		beta = beta*10

	return thetas 



