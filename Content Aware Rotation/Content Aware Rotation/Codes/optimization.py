import numpy as np 
import matplotlib.pyplot as plt 
from line_matrix import *
from shape_matrix import *

def fix_theta_solve_v(line,shape,boundary,b,lambda_l,lambda_s,lambda_b):
	"""
		This is a Quadratic in V and can be solved by using 
		sparse linear equations. Quadratic Optimization problem.
	"""
	print('Solving the sparse linear system of equations now...')
	A = shape + lambda_b*(np.transpose(boundary).dot(boundary)) + lambda_l*line
	b = -lambda_b*(np.transpose(boundary).dot(b))
	V = np.linalg.solve(A,b)
	return V


	
def fix_v_solve_theta(parameters):
	"""
		This is a non trivial function to solve. Use the half
		quadratic method to solve this equation.
	"""
	beta_min = 10
	beta_max = 10000
	beta = beta_min

	while(beta<=beta_max):

		"""
			Fix phi, update theta. Quadratic in Theta and can be
			solved by using linear system of equations.
		"""

		"""
			Fix Theta, update phi. This is a single variable equation
			and is solved by constant lookups, though gradient descent 
			can be used.
		"""

		beta = beta*10

	return thetas 



