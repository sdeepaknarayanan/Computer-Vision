
## Implementation of Difference of Gaussian Filter

import numpy as np 
import cv2 as cv
import math
import matplotlib.pyplot as plt 
from decimal import Decimal


# Initialise both the filters to zero
dog_1 = np.zeros((11,11))
dog_2 = np.zeros((11,11))
diff_of_gauss = np.zeros((11,11))

# Define the Sigma's
sigma_1 = 2.5
sigma_2 = 1.5

# Define the Gaussian with sigma included ...
def gaussian(x,y,sigma):
	return ((1/(2*math.pi*sigma**2))*math.exp(-(x**2 + y**2)/(2*sigma**2)))

image = cv.imread('img3.jpg')
image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
print('The dimensions of the image are :',image.shape)

#Image resizing to make processing faster
if image.shape[0]>1500 and image.shape[1]>750:
	image = cv.resize(image,None, fx = 0.30, fy = 0.30, interpolation = cv.INTER_CUBIC);
	print("The image is big -- resized image has dimensions ",image.shape)


# Initialising the Gaussian Filters, before taking their subtraction

for i in range(11):
    for j in range(11):
        dog_1[i][j] = gaussian(i-5, j-5, 2.5)
        dog_2[i][j] = gaussian(i-5, j-5, 1.5)

# Set the filters to the specified data format
dog_1 = dog_1.astype(np.float32)
dog_2 = dog_2.astype(np.float32)


# The DoG Filter is initialised here .....

for i in range(11):
    for j in range(11):
        diff_of_gauss[i][j] = dog_1[i][j] - dog_2[i][j]

# Set the filter to the specified Data Format
diff_of_gauss = diff_of_gauss.astype(np.float32)

# Storing the DoG in a File
file = open('DoG Filter.txt','w')
lst = []
for i in range(11):
	for j in range(11):
		lst.append('%.4E'%Decimal(str(diff_of_gauss[i][j])))
	for k in lst:
		file.write(k+ '  ')
	file.write('\n\n')
	lst = []
file.close()

# Initializing a 2D Array to accommodate for zero padding for the corner pixels..

shap = (image.shape[0]+10, image.shape[1]+10)
img_ = np.zeros(shap)
temp = 0
# Make sure that this has the value of our original image, with padded zeros
for i in range(5,shap[0]-5):
	for j in range(5, shap[1]-5):
		img_[i][j]  = image[i-5][j-5]

# Final Image to be shown or saved ... Both are done in this implementation
image_dog = np.zeros(image.shape)
image_dog = image_dog.astype(np.float32)

# Convolution of the Image 
for i1 in range(5,shap[0]-5):
	for i2 in range(5,shap[1]-5):
		for i in range(11):
			for j in range(11):
				temp+= img_[i1-5+i][i2-5+j]*diff_of_gauss[i][j]
		image_dog[i1-5][i2-5] = temp
		temp = 0

image_w = image_dog.astype(np.float32)
# Display the image
plt.figure(figsize = (10,10))
plt.title(' The Image after applying DoG')
plt.imshow(image_dog, cmap = 'gray')
plt.show()

# Saving the image after the Difference of Gaussian Filter is Performed
cv.imwrite('Diff_of_Gauss.jpg',image_w)

# Now we generate a binary image after perfoming the DOG Operation

image_final = np.zeros(image.shape)
image_final = image_final.astype(np.float32)
for i in range((image.shape[0])):
	for j in range((image.shape[1])):
		try:
			if image_dog[i][j]*image_dog[i+1][j]<=0 or image_dog[i][j]*image_dog[i-1][j]<=0 or image_dog[i][j]*image_dog[i][j+1]<=0 or image_dog[i][j]*image_dog[i][j-1]<=0 :
				image_final[i][j] = 255
                
		except:
			continue

image_final = image_final.astype(np.uint8)
# Display the Binary Image 
plt.figure(figsize = (10,10))
plt.title('Binary Image')
plt.imshow(image_final, cmap = 'gray')
plt.show()

#Save the Binary Image
cv.imwrite('Binary_Diff_of_Gauss.jpg', image_final)
