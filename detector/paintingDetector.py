from edges import getSubsetWithEdgeAnalysis
from featureDetection import runFeature
import numpy as np
import glob
from matplotlib import pyplot as plt

import cv2 as cv




i = 5



imDir = "../artSamples/"

imnames = glob.glob(imDir + '*.jpg')
imnames = sorted(imnames)
# descriptors = np.empty((1,159))
# keypoints = np.empty((1,159))
descriptors = []
keypoints = []
matchList = []
sift = cv.xfeatures2d.SIFT_create()
for f in range(len(imnames)):
	img = cv.imread(imnames[f])
	kp, des = sift.detectAndCompute(img,None)
	descriptors.append(des)
	keypoints.append(kp)

inputImage = ''
t = 0
house = cv.imread('../queryImages/house-of-parliment-NotIdentical.jpg',0)
mona = cv.imread('../queryImages/mona-lisa.jpg',0)
wall = cv.imread('../queryImages/wall-clocks.jpg',0)
scream = cv.imread('../queryImages/the-scream.jpg',0)
starry = cv.imread('../queryImages/starry-night.jpg',0)
picasso = cv.imread('../queryImages/old-artist-chicago-picasso.jpg',0)

allColorScores = np.load('../color/results.npy')

if i == 0:
	img1 = house
	inputImage = 'house-of-parliment-NotIdentical'
	t = 0.30
elif i == 1:
	img1 = mona
	inputImage = 'mona-lisa'
	t = 2.00
elif i == 2:
	img1 = scream
	inputImage = 'the-scream'
	t= 0.20
elif i == 3:
	img1 = starry
	inputImage = 'starry-night'
	t = 0.03
elif i == 4:
	img1 = picasso
	inputImage = 'old-artist-chicago-picasso'
	t = 0.50
elif i == 5:
	img1 = wall
	inputImage = 'wall-clocks'
	t = 0.40

scores = np.zeros((1,159))
subsetIndxs = getSubsetWithEdgeAnalysis(inputImage,t)

featureScores = runFeature(img1, subsetIndxs, keypoints, descriptors)
colorScores = allColorScores[i]
scores[:,np.argmax(featureScores)] += .9
scores[:,np.argmax(colorScores)] += .1
resultIm = imnames[np.argmax(scores)]
name = imnames[np.argmax(scores)][14:-4]
artist = ""
date = ""
style = ""

if name == "houses-of-parliament":
	name = "Houses of Parliament"
	artist = "Claude Monet"
	date = "1904"
	style = "Impressionism"
elif name == "mona-lisa":
	name = "Mona Lisa"
	artist = "Leonardo da Vinci"
	date = "1504"
	style = "High Renaissance"
elif name == "the-scream-1893":
	name = "The Scream"
	artist = "Edvard Munch"
	date = "1893"
	style = "Expressionism"
elif name == "the-starry-night":
	name = "The Starry Night"
	artist = "Vincent van Gogh"
	date = "1889"
	style = "Post-Impressionism"
elif name == "old-guitarist-chicago":
	name = "The Old Blind Guitarist"
	artist = "Pablo Picasso"
	date = "1903"
	style = "Expressionism"
elif name == "the-persistence-of-memory-1931":
	name = "The Persisence of Memory"
	artist = "Salvador Dali"
	date = "1931"
	style = "Surrealism"
name = ("Name: " + name + "  ")
artist = ("Artist: " + artist + "  ")
date = ("Date: " + date + "  ")
style = ("Style: " + style + "  ")

im = cv.imread(resultIm)
cv.imshow(name + artist + date + style, im)

wait = raw_input("Hit any buttong to continue")




