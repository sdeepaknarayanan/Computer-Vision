import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random 

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
		threshold = 10
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

def warpimage(image1, image2,dimgf, x_offset, y_offset,x,y,H,flag ):

	img1 = image1
	img2 = image2
			
	img_temp = np.zeros((x,y,3))
	if flag==1:		
		for elem in range(len(H)):
			for i in range(img1.shape[0]):
				for j in range(img2.shape[1]):
					if H[elem][0] == dimgf[i][j]:
						computed = H[elem][1].dot(np.asarray([j,i,1]).T)
						computed = computed/computed[2]
						computed[0] = int(computed[0])
						computed[1] = int(computed[1])
						x_c = int(computed[0])
						y_c = int(computed[1])
						try:
						    for i1 in range(-2,3):
						    	for i2 in range(-2,3):
						    	    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img2[i,j]
						except:
							continue
					else:
						computed = H[0][1].dot(np.asarray([j,i,1]).T)
						computed = computed/computed[2]
						computed[0] = int(computed[0])
						computed[1] = int(computed[1])
						x_c = int(computed[0])
						y_c = int(computed[1])
						try:
						    for i1 in range(-2,3):
						    	for i2 in range(-2,3):
						    	    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img2[i,j]
						except:
							continue
	else:

		for i in range(img1.shape[0]):
			for j in range(img1.shape[1]):
				img_temp[i+x_offset][j+y_offset] = img1[i][j]
			for i in range(img1.shape[0]):
				for j in range(img2.shape[1]):
					computed = H.dot(np.asarray([j,i,1]).T)
					computed = computed/computed[2]
					computed[0] = int(computed[0])
					computed[1] = int(computed[1])
					x_c = int(computed[0])
					y_c = int(computed[1])
					try:
						for i1 in range(-2,3):
						    for i2 in range(-2,3):
						   	    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = img2[i,j]
					except:
						continue

	img_temp = img_temp.astype(np.uint8)
	return img_temp

def quantize(depth_image, quantum_count):
	image = np.zeros(depth_image.shape)
	for i in range(depth_image.shape[0]):
		for j in range(depth_image.shape[1]):
			temp = int(depth_image[i][j]/25)
			image[i][j] = temp
	return image


img1 = cv.imread('RGBD dataset/000000292/im_0.jpg')
img2 = cv.imread('RGBD dataset/000000292/im_1.jpg')
poc, points1, points2 = begin(img1, img2)
dimg2 = cv.imread('RGBD dataset/000000292/depth_1.jpg')
dimg2 = cv.cvtColor(dimg2, cv.COLOR_BGR2GRAY)
dimgf = quantize(dimg2,11)

hom_list = [[] for i in range(11)]
print(" Press a key using waitKey function....")

print(" Quantising and choosing - Number of levels = 11")
for elem in range(len(poc)):
		for i in range(dimg2.shape[0]):
			for j in range(dimg2.shape[1]):
				if points2[elem][0]==i and points2[elem][1]==j:
					hom_list[int(dimg2[i][j]//25)].append(poc[elem])
print(" Number of points matching in each level... ")
for i in hom_list:
	print(len(i))

H = []
highest = -1
for i in range(11):
	best_so_far = 0
	best_H = np.eye(3)
	if len(hom_list[i])>4:
		highest = max(highest, i)

		H.append((i,ransac(np.asarray(hom_list[i]))))
	else:
		H.append((i,H[highest][1]))
"""
image = warpimage(img1, img2,dimgf, 5000,5000,10000,10000,H,1)
cv.imwrite('000000292_with_depth.jpg',image)
""" 

def imagewarp(img1,img2,H):
	img_tmp = np.zeros((2000,2000))
	x_offset = 500
	y_offset = 500
	for i in range(img1.shape[0]):
	    for j in range(img1.shape[1]):
	        computed = H.dot(np.asarray([j,i,1]).T)
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
	return img_tmp

H = ransac(poc)
image = imagewarp(img1,img2,H)
print(" RANSAC DONE....")
cv.imwrite("In_built_1.jpg",image)