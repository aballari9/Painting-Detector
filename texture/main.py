
# OpenCV bindings
import cv2
# To performing path manipulations
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import csv
import matplotlib.pyplot as plt

import numpy as np

###########################
# CODE FROM: http://hanzratech.in/2015/05/30/local-binary-patterns.html
###########################

# List for storing the LBP Histograms, address of images and the corresponding label

train_images = cvutils.imlist("input/")

# List for storing the LBP Histograms, address of images and the corresponding label
X_test = []
X_name = []
# y_test = []

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test

import scipy.misc

for radius in [3]:
    print radius

    for train_image in train_images:
        # Read the image
        im = cv2.imread(train_image)
        im = scipy.misc.imresize(im, (1057, 756))
        scipy.misc.imsave(train_image, im)
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # radius = 3
        # Number of points to be considered as neighbourers
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = x[:, 1]/sum(x[:, 1])
        # Append image path in X_name
        X_name.append(train_image)
        # Append histogram to X_name
        X_test.append(hist)
        # Append class label in y_test
        # y_test.append(train_dic[os.path.split(train_image)[1]])

    # Display the training images
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows,ncols)
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col].imshow(cv2.cvtColor(cv2.imread(X_name[row*ncols+col]), cv2.COLOR_BGR2RGB))
            axes[row][col].axis('off')
            axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))

    plt.show()

    # Store the path of testing images in test_images
    test_images = cvutils.imlist("test/")

    for test_image in test_images:
         # Read the image
        im = cv2.imread(test_image)
        im = scipy.misc.imresize(im, (1057, 756))
        scipy.misc.imsave(test_image, im)
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # radius = 3
        # Number of points to be considered as neighbourers
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = x[:, 1]/sum(x[:, 1])
        # Display the query image
        cvutils.imshow("** Query Image -> {}**".format(test_image), im)
        plt.show()
        results = []
        # For each image in the training dataset
        # Calculate the chi-squared distance and the sort the values
        for index, x in enumerate(X_test):
            score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.HISTCMP_CHISQR)
            results.append((X_name[index], round(score, 3)))
        results = sorted(results, key=lambda score: score[1])
        # Display the results
        fig, axes = plt.subplots(nrows,ncols)
        fig.suptitle("** Scores for -> {}**".format(test_image))
        for row in range(nrows):
            for col in range(ncols):
                axes[row][col].imshow(cv2.cvtColor(cv2.imread(results[row*ncols+col][0]), cv2.COLOR_BGR2RGB))
                axes[row][col].axis('off')
                axes[row][col].set_title("Score {}".format(results[row*ncols+col][1]))
        plt.show()
