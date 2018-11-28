import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import pickle


def runFeature(img1, indx, keypoints, descriptors):


	# np.save("keypoints.npy",keypoints)
	# np.save("descriptors.npy",descriptors)

	# pickle.dump(descriptors,open('descriptors.pkl', 'wb'))
	# pickle.dump(keypoints,open('keypoints.pkl', 'wb'))

	# descriptors = pickle.load(open('descriptors.pkl', 'rb'))
	# keypoints = pickle.load(open('keypoints.pkl', 'rb'))

	# keypoints = np.load("keypoints.npy",keypoints)
	# descriptors = np.load("descriptors.npy",descriptors)
	sift = cv.xfeatures2d.SIFT_create()

	scores = np.empty((1,159),dtype = 'uint16')
	for f in range(159):
		# Initiate SIFT detector
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = keypoints[f], descriptors[f]
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k = 2)
		# Need to draw only good matches, so create a mask
		matchesMask = [[0,0] for g in xrange(len(matches))]
		# ratio test as per Lowe's paper
		countGood = 0
		for g,(m,n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				countGood += 1
				matchesMask[g]=[1,0]
			# draw_params = dict(matchColor = (0,255,0),
		#                    singlePointColor = (255,0,0),
		#                    matchesMask = matchesMask,
		#                    flags = 0)
		scores[:,f] = countGood
		# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
		# plt.imshow(img3,),plt.show()
	return scores
	