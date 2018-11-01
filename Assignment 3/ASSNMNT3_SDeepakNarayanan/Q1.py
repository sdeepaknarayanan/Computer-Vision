import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random 

## Initialisation of the Images All Inbuilt Functions to generate matches and every other required things.....
def begin(img1, img2):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches=sorted(matches, key= lambda x:x.distance)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
    cv.imshow("Matching Key Points",img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
    poc = []
    for elem in matches:
        im_1_index = (elem.queryIdx)
        im_2_index = (elem.trainIdx)
        x_1 = kp1[im_1_index].pt
        x_1_ = kp2[im_2_index].pt
        poc.append((x_1,x_1_))
    points1 = []
    points2 = []
    for elem in matches:
        img1_ = elem.queryIdx
        img2_ = elem.trainIdx
        (x1,y1) = kp1[img1_].pt
        (x2,y2) = kp2[img2_].pt
        points1.append((x1, y1))
        points2.append((x2, y2))
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    return poc,points1,points2

# Main Function for Computing Homography for N points.......
def homography(poc):
    # poc - points of correspondences in the matches returned.....
    number_of_rows = 2*len(poc)
    number_of_cols = 9
    A = np.zeros((number_of_rows,number_of_cols))
    for i in range(len(poc)):

        x_1 = (poc[i][0][0])
        y_1 = (poc[i][0][1])
        z_1 = 1 # wi
        x_2 = (poc[i][1][0])
        y_2 = (poc[i][1][1])
        z_2 = 1 #wi'
        A[2*i,:] = [x_1,y_1,1,0,0,0,-x_2*x_1,-x_2*y_1,-x_2*z_1]
        A[2*i+1,:]=[0,0,0,x_1,y_1,1,-y_2*x_1,-y_2*y_1,-y_2*z_1]
    
    U, D, V_t = np.linalg.svd(A)
    h = V_t[-1]
    H = np.zeros((3,3))
    H[0] = h[:3]
    H[1] = h[3:6]
    H[2] = h[6:9]
    H = H/H[2,2]
    val = H.dot([poc[1][0][0],poc[1][0][1],1])
    val = val/val[2]
    val = val.astype(np.uint32)
    return H

## RANSAC Algorithm
def ransac(poc):
    best_so_far = 0
    best_H = 0
    for i in range(10000):
        maybeinliers = []
        index = random.sample(range(1,len(poc)),4)
        matches = [poc[i] for i in index]
        H = homography(matches)
        threshold = 2
        count = 0
        for elem in range(len(poc)):
            curr = poc[elem][0]
            new_elem = np.asarray([curr[0],curr[1],1]).T
            transformed = H.dot(new_elem)
            transformed = transformed/transformed[2]
            act = np.asarray([poc[elem][1][0],poc[elem][1][1],1])
            if np.linalg.norm(transformed-act)<=threshold:
                count+=1
                maybeinliers.append(poc[elem])
        if count>best_so_far:
            best_so_far = count
            best_H = homography(maybeinliers)
    return best_H

# Image Warping is Done Here - We compute the Homographies as follows: 
# Homography of 2 wrt 1; We never warp 1.... We warp 2 wrt 1, then 3 wrt 2 and multiply by 2 wrt 1 to maintain the offset;;
# Then we maintain the same by computing homography of 4 wrt 3 and then multiplying by 3 wrt 2 and by 2 wrt 1..... For overall
def warpimage(image1, image2, image3, image4,  x_offset, y_offset,x,y,H12,H13,H14):

	img1 = image1
	img2 = image2
	img3 = image3
	img4 = image4
	        
	img_temp = np.zeros((x,y,3))

	for i in range(img1.shape[0]):
	    for j in range(img1.shape[1]):
	        img_temp[i+x_offset][j+y_offset] = img1[i][j]

	for i in range(img1.shape[0]):
	    for j in range(img1.shape[1]):
	        computed = H12.dot(np.asarray([j,i,1]).T)
	        computed = computed/computed[2]
	        computed[0] = int(computed[0])
	        computed[1] = int(computed[1])
	        x_c = int(computed[0])
	        y_c = int(computed[1])
	        try:
	            for i1 in range(-1,2):
	                for i2 in range(-1,2):
	                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img2[i,j]
	        except:
	            continue
	        
	for i in range(img1.shape[0]):
	    for j in range(img1.shape[1]):
	        computed = H13.dot(np.asarray([j,i,1]).T)
	        computed = computed/computed[2]
	        computed[0] = int(computed[0])
	        computed[1] = int(computed[1])
	        x_c = int(computed[0])
	        y_c = int(computed[1])
	        try:
	            for i1 in range(-1,2):
	                for i2 in range(-1,2):
	                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img3[i,j]
	        except:
	            continue

	for i in range(img1.shape[0]):
	    for j in range(img1.shape[1]):
	        computed = H14.dot(np.asarray([j,i,1]).T)
	        computed = computed/computed[2]
	        computed[0] = int(computed[0])
	        computed[1] = int(computed[1])
	        x_c = int(computed[0])
	        y_c = int(computed[1])
	        try:
	            for i1 in range(-2,3):
	                for i2 in range(-2,3):
	                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img4[i,j]
	        except:
	            continue

	img_temp = img_temp.astype(np.uint8)

	return img_temp

        
# Compute the Homography of 1 and 2;
# Compute the Homography of 3 and 4;
# Compute the Homography of 1,2 and 3,4
# That'll be the final image. 

image = ['Images_Asgnmt3_1/I1/STA_0031.JPG',
'Images_Asgnmt3_1/I1/STB_0032.JPG',
'Images_Asgnmt3_1/I1/STC_0033.JPG',
'Images_Asgnmt3_1/I1/STD_0034.JPG']

## By Default this is the image list that we have......
print(" Images reading next' press enter... we're using cv.waitKey(0) - do not close.....")

## Reading the Images 
img1 = cv.imread(image[0])
img2 = cv.imread(image[1])
img3 = cv.imread(image[2])
img4 = cv.imread(image[3])

print("Resizing all the images ")
img1 = cv.resize(img1,(800,532))
img2 = cv.resize(img2,(800,532))
img3 = cv.resize(img3,(800,532))
img4 = cv.resize(img4,(800,532))

print(" Doing the initially needed things .... computing all the matching points...")
## Computing the Points of Matches to be used in RANSAC
poc21,points2_,points1_ = begin(img2,img1)
poc32,points3,points2 = begin(img3, img2)
poc43,points4_,points3_ = begin(img4,img3)
## I have commented the code for inbuilt homography -- we can use it when we need to predict what's needed
"""
print(" RANSAC now ...")
H12 = ransac(poc21)
H13 = H12.dot(ransac(poc32))
H14 = H13.dot(ransac(poc43))
"""
H12 = cv.findHomography(points2_,points1_,cv.RANSAC,4)[0]
H13 = H12.dot(cv.findHomography(points3,points2,cv.RANSAC,4)[0])
H14 = H13.dot(cv.findHomography(points4_,points3_,cv.RANSAC,4)[0])
print(" Warping on the way...")
# Function for Image Warping; Storing Image later below.....
image_temp = warpimage(img1,img2,img3,img4,5000,5000,10000,10000,H12,H13,H14)
print(" Writing the Image obtained.....")

cv.imwrite('Panorama_I1_IBH.jpg',image_temp)

