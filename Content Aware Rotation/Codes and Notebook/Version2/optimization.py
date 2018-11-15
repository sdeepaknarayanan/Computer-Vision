import numpy as np 
import matplotlib.pyplot as plt 
import math
from line_matrix import *
from shape_matrix import *
from extractlines import quantize_and_get
from boundary_matrix import *
from pk import *
from math import cos, sin

def fix_theta_solve_v(line,shape,boundary,b,lambda_l,lambda_r,lambda_b,n,k):

    """
        This is a Quadratic in V and can be solved by using 
        sparse linear equations. I manually write down all the equations and differentiate
        each of them. The equations that I found are only being solved here.
        1. The first equation solves for all the vertices that were optimised, keeping theta a constant.
        2. The second one solves for all the thetas, now that the vertices have been solved.
    """

    print('\nTheta is a constant; Solving for V .')
    A = shape/n + lambda_b*(np.transpose(boundary).dot(boundary)) + lambda_l*line/k
    b = -lambda_b*(np.transpose(boundary).dot(b))
    V = np.linalg.solve(A,b)
    print('\n V is done. Now solving thetas with the obtained V')
    return V



def fix_v_solve_theta(UK,lines,thetas,V_new,rotation_angle,dx,dy,N,x,y,sdelta,lambda_l,lambda_r):

    """
        We employ a half quadratic splitting method here. 
        We basically warm up the betas, as the authors call it, 
        to a higher value, slowly putting in harder constraints on the 
        rotation possible. This ensures that the lines are rotated only
        minimally from the bins to which they belong.
    """

    """ 
    Standard Initialisations are being done here
    """

    k = len(lines)
    delta = rotation_angle

    """
        Beta, initially set to 1 here. It is slowly
        increased to 10000
    """

    beta_min = 1
    beta_max = 10000
    beta = beta_min

    M = len(thetas)
    """
        Using the notation that the authors use for the number 
        of bins. We put each quad in one of these bins and then
        rotate, accordingly.
    """

    A = np.zeros((k,M))

    """
        This variable A has all the corresponding bins
        for all the lines that were detected using the LSD.
        Assigning the values below.
    """

    for i in range(len(lines)):
        temp=  lines[i][5]
        A[i][int(temp)-1] = 1

    T_1 = np.zeros((M,M))
    T_1[0:M-1,1:M] = np.eye(M-1)
    T_1[M-1,0] = 1
    D = np.eye(M,M) - T_1
    Pk = computepk(lines,dx,dy,N,x,y)
    Vx = np.zeros(N)
    Vy = np.zeros(N)
    for i in range(N):
        Vx[i] = V_new[2*i]
        Vy[i] = V_new[2*i+1]      
    e_x = Pk.dot(Vx)
    e_y = Pk.dot(Vy)
    e = np.array([e_x,e_y,np.arccos(np.divide(e_x,(np.sqrt(np.square(e_x)+np.square(e_y)))))*180/np.pi])
    e = e.transpose()
    phi_k = np.zeros(k)
    phi = np.zeros((k,100))
    print('\n Solving for thetas, V is a constant. We make this a function of phi and theta ...')
    while(beta<=beta_max):

        """
            Fix Theta, update phi. This is a single variable equation
            and is solved by constant lookups, though gradient descent 
            can be used.
        """
        for i in range(len(lines)):
            
            """
                Instead of performing Gradient Descent to solve for phi,
                we discretise the range of values that phi can take into
                100 bins. We check these bins to look for that phi that
                minimizes the optimisation function.
            """
            ek_uk = e[i][2] - lines[i][4]   # Angle between line and bin
            bin_ = lines[i][5]              # Bin for the line in question
            thetam_k = thetas[int(bin_)-1]  # Computing the angle if beta->infintiy

            phi[i] = (np.linspace(ek_uk, thetam_k, num= 100))
            temp = np.zeros(100)
            U = UK[i]     

            for j in range(100):
                Rk = np.array([[cos(phi[i][j]/np.pi*180),-sin(phi[i][j]/np.pi*180)],
                    [sin(phi[i][j]/np.pi*180), cos(phi[i][j]/np.pi*180)]])

                current = (Rk.dot(U).dot(np.transpose(Rk)) - np.eye(2)).dot((e[i][0:2]))

                temp[j] = lambda_l/k*(sum(np.square(current))) + beta*(np.square(phi[i][j] - thetam_k)).sum()

            index = np.argmin(temp)

            phi_k[i] = phi[i][index]    # Find that phi that minimizes the energy by creating a small table

        """
            Fix phi, update theta. Quadratic in Theta and can be
            solved by using linear system of equations. Form the quadratic,
            differentiate with respect to theta. This is convex and is a
            function of phi and theta
        """

        H = beta*(np.transpose(A).dot(A)) + lambda_r*np.diag(sdelta) + lambda_r*np.transpose(D).dot(D)
        h = beta*np.transpose(A).dot(phi_k) + (lambda_r*np.diag(sdelta).dot(np.ones(M))*delta)
        thetas = np.linalg.solve(H,h)
        beta = beta*10
    print('\n Half Quadratic splitting optimisation done ..., solved for V and theta.\n')
    return thetas