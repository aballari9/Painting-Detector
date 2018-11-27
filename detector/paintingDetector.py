from edges import getSubsetWithEdgeAnalysis
from featureDetection import runFeature
import numpy as np
import sys
sys.path.insert(0, '../color/')


house = cv.imread('../queryImages/house-of-parliment-NotIdentical.jpg',0)
mona = cv.imread('../queryImages/mona-lisa.jpg',0)
wall = cv.imread('../queryImages/wall-clocks.jpeg',0)
scream = cv.imread('../queryImages/the-scream.jpeg',0)
starry = cv.imread('../queryImages/starry-night.jpeg',0)
picasso = cv.imread('../queryImages/old-artist-chicago-picasso.jpg',0)
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
for i in range(6):

	colorResults = np.load('results.npy')
	subsetIndxs = 