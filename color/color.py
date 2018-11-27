# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
from scipy.misc import imread

hists = np.load("histograms.npy")
queryHists = np.load("queryHistograms.npy")

imageFiles = np.load("imageFiles.npy")
queryFiles = np.load("query_images.npy")

# argmin
# method = cv2.HISTCMP_BHATTACHARYYA

#argmax
method = cv2.HISTCMP_INTERSECT
# method = cv2.cv2.HISTCMP_CORREL

# initialize the results dictionary
results = np.zeros((queryHists.shape[0],hists.shape[0]), np.float32)

# for i in range(1):
for i in range(queryHists.shape[0]):
    for j in range(hists.shape[0]):
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(queryHists[i], hists[j], method)
        results[i][j] = d

        if "parliament" in imageFiles[j]:
            print d


    plt.imshow(imread(queryFiles[i]))
    plt.show()

    i_max = np.argmax(results[i])
    print results[i]
    print results[i][i_max]
    img = imageFiles[i_max]
    plt.imshow(imread(img))
    plt.show()

    print "next"

print results.shape

np.save("results.npy", results)

# sort the results
# results = sorted([(v, k) for (k, v) in results.items()], reverse = True)

# # show the query image
# fig = plt.figure("Query")
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(images["monaLisa_museum.jpg"])
# plt.axis("off")

# # initialize the results figure
# fig = plt.figure("Results: %s" % (methodName))
# fig.suptitle(methodName, fontsize = 20)

# # loop over the results
# for (i, (v, k)) in enumerate(results):
#     # show the result
#     ax = fig.add_subplot(1, len(images), i + 1)
#     ax.set_title("%s: %.2f" % (k, v))
#     plt.imshow(images[k])
#     plt.axis("off")

# # show the OpenCV methods
# plt.show()
