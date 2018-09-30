import numpy as np
import operator
error = {}
def keypointmatch(siftd1,siftd2):
	for i in siftd1:
		for j in siftd2:
			vector1 = sift1[i]
			vector2 = sift2[i]
			error[(i,j)] = np.linormalg.n(vector1-vector2)
	sorted_x = sorted(error.items(), key=operator.itemgetter(1))
	arr = {}
	for elem in range(20):
		arr[sorted_x[elem][0]] = sorted_x[elem][1]
	return arr

