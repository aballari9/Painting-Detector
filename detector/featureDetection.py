import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import pickle


def runFeature():
	imDir = "../artSamples/"

	imnames = glob.glob(imDir + '*.jpg')
	imnames = sorted(imnames)

	# descriptors = np.empty((1,159))
	# keypoints = np.empty((1,159))
	descriptors = []
	keypoints = []
	allScores = np.empty((6,159),dtype = 'uint16')
	matchList = []
	sift = cv.xfeatures2d.SIFT_create()
	for i in range(len(imnames)):
		img = cv.imread(imnames[i])
		kp, des = sift.detectAndCompute(img,None)
		descriptors.append(des)
		keypoints.append(kp)

	# np.save("keypoints.npy",keypoints)
	# np.save("descriptors.npy",descriptors)

	# pickle.dump(descriptors,open('descriptors.pkl', 'wb'))
	# pickle.dump(keypoints,open('keypoints.pkl', 'wb'))

	# descriptors = pickle.load(open('descriptors.pkl', 'rb'))
	# keypoints = pickle.load(open('keypoints.pkl', 'rb'))

	# keypoints = np.load("keypoints.npy",keypoints)
	# descriptors = np.load("descriptors.npy",descriptors)

	house = cv.imread('../queryImages/house-of-parliment-NotIdentical.jpg',0)
	mona = cv.imread('../queryImages/mona-lisa.jpg',0)
	wall = cv.imread('../queryImages/wall-clocks.jpeg',0)
	scream = cv.imread('../queryImages/the-scream.jpeg',0)
	starry = cv.imread('../queryImages/starry-night.jpeg',0)
	picasso = cv.imread('../queryImages/old-artist-chicago-picasso.jpg',0)

	for i in range(0,6):
		if i == 0:
			img1 = house
		elif i == 1:
			img1 = mona
		elif i == 2:
			img1 = wall
		elif i == 3:
			img1 = scream
		elif i == 4:
			img1 = starry
		elif i == 5:
			img1 = picasso

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
		allScores[i,:] = scores[:]
		matchList.append(imnames[np.argmax(allScores[i])])
	return matchList

print(runFeature())
	