""" 

Content Aware Rotation, In ICCV 2013	
Authors: Kaiming He, Huiwen Chan, Jian Sun

An implementation by S Deepak Narayanan, Indian Institute of Technology, Gandhinagar

"""


""" 
Standard Imports for the rest of the program 
"""

import matlplotlib.pyplot as pyplot
import cv2 as cv2
import numpy as np 
import random as rd 
import lsd
import meshgrid
import energyfunction
import optimization
import eucompute
import 
"""
Initially we read the Image to be rotated. Rotation Angle fixed
"""
image = cv.imread("")



"""
Line Segment Detection is the first step;
Calling the function for the same
"""
bins = lsd.detect(image)

"""
Detected and Quantised the lines;
We have put them into discrete bins
"""

"""
Displaying the Initial MeshGrid to be warped
"""
meshgrid.display(image)

"""
Defining the energy function
"""
