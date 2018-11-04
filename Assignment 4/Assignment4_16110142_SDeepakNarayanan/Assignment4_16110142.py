"""
Assignment 4 3D Computer Vision 
Submitted by S Deepak Narayanan
16110142, B.Tech, CSE
"""



# Standard Imports are being done here
import cv2 as cv
import random
import numpy as np
import matplotlib.pyplot as plt
print('Standard Imports are over \n')




# Standard Image Reads and Display
print('Please Press Enter twice. I am using cv.imshow() function here.\n')
img1 = cv.imread('Datasets/Vintage_Reference.png')
img2 = cv.imread('Datasets/Vintage_Source.png')
img1 = cv.resize(img1,(300,300))
img2 = cv.resize(img2,(300,300))

cv.imwrite("Reference.jpg",img1)
cv.imwrite("Source.jpg",img2)

cv.imshow("",img1)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("",img2)
cv.waitKey(0)
cv.destroyAllWindows()





print('The resized image is of size 300 x 300 and the shape is',img1.shape)





print("\nSIFT Descriptor is being used here. We perform Lowe's Ratio Test")
sift = cv.xfeatures2d.SIFT_create()





kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
des_1 = []
des_2 = []
# J.Lowe's Ratio Test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
print("SIFT Computation is over. Now, we are computing the Fundamental Matrix.\n")





# This is for the estimation of the Fundamental Matrix
# We need integral coordinates - Whatever has been obtained so far might not be integral
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2)
print("Fundamental Matrix F is \n",F)
# We select best points only, those that are actually important, remove the Outliers
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]





# Creating Keypoint List for the Images
kplist1 = []
kplist2 = []
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        kplist1.append(cv.KeyPoint(j,i,8))
        kplist2.append(cv.KeyPoint(j,i,8))
# Computing Descriptors using SIFT for all the points.
_,desc1 = sift.compute(img1, kplist1)
_,desc2 = sift.compute(img2, kplist2)
desc2 = desc2.reshape(img2.shape[0], img2.shape[1], 128)
print("Fundamental Matrix computation done. We'll be computing the EpiLines here.\n")



# Computing the points as an np.array
points = []
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        temp = np.array((j,i))
        points.append(temp)
points = np.array(points) 




lines = cv.computeCorrespondEpilines(points.reshape(-1,1,2), 1,F)
lines = lines.reshape(-1,3)
print("Epilines completed.")





print("Important section of the code is getting executed here ... \n")
print("We're currently computing the corresponding pixels... \n")
print("Image size, as mentioned above is 300 x 300. Here I'll be displaying progress as the number of iterations completed...\n")
print("The Progress goes all the way from 0 to 89000. Once it reaches 89000, we're done with computation.")
img_final = np.zeros(img1.shape)
dval = {}
# This section matches the points using the SIFT Descriptor
# We also compute the corresponding points 
# We are using precomputed Descriptor
for i in range(len(points)):
    if i%1000 == 0:
        print(i)
    try:
        curr_point = points[i]
        a,b,c = lines[i]
        potential = []
        for x in range(img1.shape[1]):
            try:
                # ymin and yman were set to have a region around 
                # Epipolar Line for our geometrical consideration.
                # The region gave worse results relatively and hence 
                # ymin == ymax
                ymin = int(-(a*x + c)/b)
                ymax = int(-(a*x + c)/b)
                val_range = range(ymin, ymax+1)
                for y in val_range:
                    if y>=0 and y<img1.shape[0]:
                        potential.append((x,int(y)))
            except:
                break
        pt_des = desc1[i]
        al_des = []
        for i1 in range(len(potential)):
            al_des.append(desc2[potential[i1][1]][potential[i1][0]])
        min_norm = np.inf
        min_index = -1
        
        for val in range(len(al_des)):
            temp = np.linalg.norm(pt_des - al_des[val])
            if temp<min_norm:
                min_norm = temp
                min_index = val
        dval[tuple(curr_point)] = potential[min_index]
    except:
        pass




# Evaluating the final new image.

print("Done with Computation! Computing the new image. ")
temp= np.zeros(img1.shape)
for elem in dval:
    try:
        temp[(elem[1],elem[0])] = img2[(dval[elem][1],dval[elem][0])]
    except:
        pass
temp = np.uint8(temp)
# Data Type of the final new image is being changed to what it is supposed to be

print("Please Press Enter -- Don't close :). I am using cv.imwait().")
cv.imshow("Final Image",temp)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("Result.jpg",temp)
