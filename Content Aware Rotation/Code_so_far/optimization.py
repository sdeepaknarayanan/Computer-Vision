import numpy as np 
import matplotlib.pyplot as plt 

def fix_theta_solve_v(parameters):
	"""
		This is a Quadratic in V and can be solved by using 
		sparse linear equations.
	"""



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
			solved by using linear equations.
		"""

		"""
			Fix Theta, update phi. This is a single variable equation
			and is solved by constant lookups, though gradient descent 
			can be used
		"""

		beta = beta*10

	return thetas 



