# @ Author S Deepak Narayanan
# coding: utf-8

# ## Content Aware Rotation - EE645 Course Project - S Deepak Narayanan, 16110142
# 
# * Input Rotation Angle estimation, given by the authors for 8 images in their dataset.
# * Line Extraction and Quantization 
# * Energy Function
#     * Rotation Manipulation
#     * Line Preservation
#     * Shape Preservation
#     * Boundary Preservation
# * Optimize Energy Function (A linear combination of the four sub-parts) using results from Rotation Angle and Line Extraction and Quantization
#     * Fix theta, solve for V
#         * This is itself is not a trivial problem because the authors haven't given their energy function explicitly in terms of standard variables, V and Theta.
#     * Fix V, solve for theta
#         * This again is a difficult problem to solve as it involves half-quadratic splitting technique.
# * Warp the final image using the barycentric coordinates (or) bilinear interpolation.
#     * Barycentric Coordinates allows me to break every quad into two triangles and then deform the image. 
#     * Any point inside the triangle is a normalized linear combination of the coordinates of the triangle.
# * Results 

# In[1]:


import cv2 as cv
import math
import numpy as np


# In[2]:


#Reading and Displaying the Image
image = cv.imread('Dataset/image1.png')
cv.imshow('image.jpg',image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.LineSegmentDetector()


# In[3]:


## Line Segement Detector 
# LSD has the Line Segment Detector, using OpenCV's inbuilt function. 
lsd = (cv.LineSegmentDetector(image))
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
img = grayscale
lsd = cv.createLineSegmentDetector(0)
lines = lsd.detect(img)[0] 
drawn_img = lsd.drawSegments(img,lines)


# In[4]:


lines
# Displaying the Line Segments Detected in the Image. 
# This particular part of the Code is showing the coordinates of the Line Segment ((x1,y1),(x2,y2))


# In[5]:


cv.imshow("Image after detecting line segments",drawn_img)
cv.waitKey(0)
cv.destroyAllWindows()


# Here rotated is the angle by which the Image should be rotated so that the current detected line segments, after rotation have angles in the range 0 to 180

# In[6]:


rotated = -6.1 # Here Rotated refers to the angle to be rotated of the Image.
orientations = [rotated + i for i in range(0,181,2)]
print(orientations)


# In[7]:


# Assigning the orientation for each line that we have found so far. 
# We are essentially ensuring that the orientations all fall within 0 to 180 for our problem.
line_orientation = []
for i in range(len(lines)):
    #print(lines[i])
    if lines[i][0][2]>=lines[i][0][0] and lines[i][0][3]>=lines[i][0][1]:
        line_orientation.append(math.atan((lines[i][0][2] - lines[i][0][0])/(lines[i][0][3] - lines[i][0][1]))*180/math.pi)
    if lines[i][0][2]<=lines[i][0][0] and lines[i][0][3]>=lines[i][0][1]:
        line_orientation.append(180+math.atan((lines[i][0][2] - lines[i][0][0])/(lines[i][0][3] - lines[i][0][1]))*180/math.pi)
    if lines[i][0][2]<=lines[i][0][0] and lines[i][0][3]<=lines[i][0][1]:
        line_orientation.append(math.atan((lines[i][0][2] - lines[i][0][0])/(lines[i][0][3] - lines[i][0][1]))*180/math.pi)
    if lines[i][0][2]>=lines[i][0][0] and lines[i][0][3]<=lines[i][0][1]:
        line_orientation.append(180 +math.atan((lines[i][0][2] - lines[i][0][0])/(lines[i][0][3] - lines[i][0][1]))*180/math.pi)
    if line_orientation[i]>=180:
        line_orientation[i]-=180
line_orientation = np.asarray(line_orientation)
print("The maximum line orientation and the minimum line orientation angles are ",line_orientation.max(),line_orientation.min())


# In[8]:


## Putting all the Lines into various different bins here. 
## This ensures that we actually have put the lines into their quantised bins. Canonical bins to be strictly
## rotated by the input angles are also mentioned. Do note that we are stored the "number" of the line, as
## well as the angle of the line to the corresponding bin.
bins = ([[] for i in range(90)])
for j in range(len(line_orientation)):
    if int(line_orientation[j]%2)==0:
        val = int(line_orientation[j])
        #print(val)
        bin_id = val//2
        bins[bin_id].append((j,line_orientation[j]))
    else:
        #print(val)
        val = int(line_orientation[j])
        val+=1
        bin_id = val//2
        if bin_id==90:
            bins[bin_id-1].append((j,line_orientation[j]))
        else:
            bins[bin_id].append((j,line_orientation[j]))


# Canonical Bins to be rotated by the rotation angle of the image with thier line orientations are displayed here. Note the Line ID with the rotation angle. This is useful in calculating the uk vector while performing line preservation optimization.

# In[9]:


print(" Canonical Bin 1 is ",bins[0])
print(" Canonical Bin 2 is ",bins[44])
print(" Canonical Bin 3 is ",bins[45])
print(" Canonical Bin 4 is ",bins[89])


# In[10]:


## Grid Coordinates
number_of_squares = 500
x = image.shape[0]
y = image.shape[1]
area_of_image = x*y
area_per_square = area_of_image/500
side_of_square = int(area_per_square**0.5)
print(" Each Grid Consists of ",side_of_square**2, "Pixels")


# In[11]:


grid_coordinates = []
for i in range(x):
    for j in range(y):
        if i*36<x and j*36<y:
            grid_coordinates.append((i*36,j*36))
for i in range(len(grid_coordinates)):
    temp = []
    temp.append(grid_coordinates[i][0])
    temp.append(grid_coordinates[i][1])
    grid_coordinates.append(np.asarray(temp))
grid_coordinates = np.asarray(grid_coordinates)


# In[12]:


# Number of Vertices as used in the programs under the Python Programs in Codes
N = len(grid_coordinates) 
print(' Number of Vertices in the Grid is ',N)


# In[13]:


# Initialisation of Theta to be 90 degrees. Note that 
# this angle rotated is specific only to the above given input image
thetas = [rotated for i in range(90)]
print(" The initialised values for the thetas ",thetas)

