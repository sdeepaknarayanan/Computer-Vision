import numpy as np
import pandas as pd

"""
    Some Notes in this Code - You are using 1 Indexed Co-Ordinates
    Need a way to fix them up sometime later.
"""

def formboundary(N,X,Y,gridX,gridY):
    energy_vx = np.zeros(N)
    energy_vy = np.zeros(N);
    energy_v = np.zeros(2*N)
    bx = np.zeros(N)
    by = np.zeros(N)
    b = np.zeros(2*N)

    """
        Initialization of all the variables as in above
    """

    tempgridX = gridX.astype(np.uint16)
    tempgridY = np.rint(gridY)

    """
        We need to construct matrices that capture the boundary vertices under consideration
    """
    
    idx = (tempgridX == 0)
    idx = idx.transpose().reshape(N)
    energy_vx[idx]=1
    bx[idx] = -1

    """
        First the Left Border was covered above.
    """

    idx = (tempgridX==X-1)
    idx = idx.transpose().reshape(N)
    energy_vx[idx] = 1
    bx[idx] = -(X)
    """
        Right Border was covered above
    """

    idx = (tempgridY==0)
    idx = idx.transpose().reshape(N)
    energy_vy[idx]=1
    by[idx] = -(1)

    """
        Top border was covered above.
    """

    idx = (tempgridY == Y-1)
    idx = idx.transpose().reshape(N)
    energy_vy[idx] = 1
    by[idx] = -(Y)

    """
        Bottom Border was covered above
    """
    

    for i in range(N):
        energy_v[2*i] = energy_vx[i]
        energy_v[2*i+1] = energy_vy[i]
        b[2*i] = bx[i]
        b[2*i+1] = by[i]

    """
        Finally, I have computed the required energy function for
        all the vertices and made a matrix out of it.
    """

    boundary_matrix = np.diag(energy_v)
    df = pd.DataFrame(boundary_matrix)
    df.to_csv('Boundary.csv')
    return boundary_matrix,b