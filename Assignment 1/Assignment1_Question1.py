import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt 
from decimal import Decimal 


# Read the image
image = cv.imread('img1.jpg')
print('The dimensions of the image are :', image.shape)

# Intialise 3 Guassians as per requirement
gauss_1 = np.zeros((9,9))
gauss_3 = np.zeros((9,9))
gauss_20= np.zeros((9,9))

# Define the Value of the Gaussian at a given x and y location
def gaussian(x,y,sigma):
	return (math.exp(-(x**2 + y**2)/(2*sigma*sigma)))

# Resize the image for faster processing. 
# Bigger Images take more time

#Image resizing to make processing faster
if image.shape[0]>1500 and image.shape[1]>750:
	image = cv.resize(image,None, fx = 0.30, fy = 0.30, interpolation = cv.INTER_CUBIC);
	print("The image is big -- resized image has dimensions ",image.shape)

# Gaussian Initialisation ( Offset is set to have peak value at the center)
for i in range(9):
	for j in range(9):
		gauss_1[i][j] = gaussian(i-4,j-4,1)
		gauss_3[i][j] = gaussian(i-4,j-4,3)
		gauss_20[i][j] = gaussian(i-4,j-4,20)

# Set the data format for the Gaussian
gauss_1 = gauss_1.astype(np.float32)
gauss_3 = gauss_3.astype(np.float32)
gauss_20 = gauss_20.astype(np.float32)

# Normalize the Gaussian
gauss_1 = gauss_1/gauss_1.sum()
gauss_3 = gauss_3/gauss_3.sum()
gauss_20 = gauss_20/gauss_20.sum()


# Reporting the Gaussian Kernel Values in a text file

## STD 1
file = open('STD_1.txt','w')
lst = []
for i in range(9):
	for j in range(9):
		lst.append('%.4E'%Decimal(str(gauss_1[i][j])))
	for k in lst:
		file.write(k+ '  ')
	file.write('\n\n')
	lst = []
file.close()

## STD 3
file = open('STD_3.txt','w')
lst = []
for i in range(9):
	for j in range(9):
		lst.append('%.4E'%Decimal(str(gauss_3[i][j])))
	for k in lst:
		file.write(k+ '  ')
	file.write('\n\n')
	lst = []
file.close()

## STD_20

file = open('STD_20.txt','w')
lst = []
for i in range(9):
	for j in range(9):
		lst.append('%.4E'%Decimal(str(gauss_20[i][j])))
	for k in lst:
		file.write(k+ '  ')
	file.write('\n\n')
	lst = []
file.close()


# Initializing a 2D Array to accommodate for zero padding for the corner pixels..
shap = (image.shape[0]+8, image.shape[1]+8, 3)
img_ = np.zeros(shap)
temp = 0

# Make sure that this has the value of our original image, with padded zeros
for i in range(4,shap[0]-4):
	for j in range(4, shap[1]-4):
		img_[i][j]  = image[i-4][j-4]

# Final Image to be shown or saved ... Both are done in this implementation
image_final = np.zeros(image.shape)
image_final = image_final.astype(np.float32)

# Convolution of the Image 

# Standard Deviation 1 

# Final Image to be shown or saved ... Both are done in this implementation
image_final = np.zeros(image.shape)
image_final = image_final.astype(np.float32)


# Convolution of the Image 
for i1 in range(4,shap[0]-4):
	for i2 in range(4,shap[1]-4):
		for k in range(3):
			for i in range(9):
				for j in range(9):
					temp+= img_[i1-4+i][i2-4+j][k]*gauss_1[i][j]
			image_final[i1-4][i2-4][k] = temp
			temp = 0
image_final = image_final.astype(np.uint8)
plt.figure(figsize = (10,10))
plt.title('With Standard Deviation 1')
plt.imshow(image_final)
plt.show()
cv.imwrite('Q1_STD_1.jpg', image_final)

# Standard Deviation 3

# Final Image to be shown or saved ... Both are done in this implementation
image_final = np.zeros(image.shape)
image_final = image_final.astype(np.float32)
temp = 0 
# Convolution of the Image 
for i1 in range(4,shap[0]-4):
	for i2 in range(4,shap[1]-4):
		for k in range(3):
			for i in range(9):
				for j in range(9):
					temp+= img_[i1-4+i][i2-4+j][k]*gauss_3[i][j]
			image_final[i1-4][i2-4][k] = temp
			temp = 0
image_final = image_final.astype(np.uint8)
plt.figure(figsize = (10,10))
plt.title('With Standard Deviation 3')
plt.imshow(image_final)
plt.show()
cv.imwrite('Q1_STD_3.jpg', image_final)

# Standard Deviation 20

# Final Image to be shown or saved ... Both are done in this implementation
image_final = np.zeros(image.shape)


# Convolution of the Image 
for i1 in range(4,shap[0]-4):
	for i2 in range(4,shap[1]-4):
		for k in range(3):
			for i in range(9):
				for j in range(9):
					temp+= img_[i1-4+i][i2-4+j][k]*gauss_20[i][j]
			image_final[i1-4][i2-4][k] = temp
			temp = 0
image_final = image_final.astype(np.uint8)
plt.figure(figsize = (10,10))
plt.title('With Standard Deviation 20')
plt.imshow(image_final)
plt.show()
cv.imwrite('Q1_STD_20.jpg', image_final)


