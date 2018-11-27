# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

image_files = sorted(glob.glob("../artSamples/*.jpg"))
hists = np.zeros((len(image_files), 512), np.float32)

# loop over the image paths
for i in range(len(image_files)):
# for imagePath in image_files:
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary

    imagePath = image_files[i]
    # print imagePath

    filename = imagePath[imagePath.rfind("/") + 1:]
    # print filename

    image = cv2.imread(imagePath)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()

    hists[i] = hist

print hists.shape
np.save("histograms.npy", hists)

np.save("imageFiles.npy", image_files)




image_files = sorted(glob.glob("../queryImages/*.jpg"))
hists = np.zeros((len(image_files), 512), np.float32)
# loop over the image paths
for i in range(len(image_files)):
# for imagePath in image_files:
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary

    imagePath = image_files[i]
    # print imagePath

    filename = imagePath[imagePath.rfind("/") + 1:]
    # print filename

    image = cv2.imread(imagePath)

    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()

    hists[i] = hist

    print hist.shape

np.save("queryHistograms.npy", hists)
print hists.shape

np.save("query_images.npy", image_files)







