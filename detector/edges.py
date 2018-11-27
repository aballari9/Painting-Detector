import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage import feature
from skimage.color import rgb2gray
import os

# inputImage = 'mona-lisa'
inputSigma = 0
ratios = {}

basedir = '../artSamples/'
images = []
for im in np.sort(os.listdir(basedir)):
    images.append(im[:-4])

num_of_images = len(images)
sigmas = [0] * num_of_images

threshold = 0.1
results = []
results_indicies = []

def getCoordinatesFromImage(img1):

    fig, ax1 = plt.subplots(1,1)
    plt.suptitle('Select Corresponding Points')

    ax1.set_title("Input Image")
    # ax1.axis('off')
    ax1.imshow(img1)

    axis1_xValues = []
    axis1_yValues = []

    # Handle Onclick
    def onclick(event):
        if event.inaxes == ax1:
            xVal = event.xdata
            yVal = event.ydata
            point = (xVal, yVal)
            plt.plot(xVal, yVal, ',')
            fig.canvas.draw()
            print 'image 1: ', point
            axis1_xValues.append(xVal)
            axis1_yValues.append(yVal)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if (len(axis1_xValues) < 4):
        print 'Must have at least 4 coresponding points.'
        return None, None

    # Store points in a 2xn numpy array
    points1 = np.zeros((2, len(axis1_xValues)))
    points1[0] = axis1_yValues
    points1[1] = axis1_xValues

    return points1

def getCroppedImage(im, corners):
    top_left = corners[:,0]
    top_right = corners[:,1]
    bottom_right = corners[:,2]
    bottom_left = corners[:,3]

    min_row = int(min(top_left[0], top_right[0]))
    max_row = int(max(bottom_right[0], bottom_left[0]))
    min_col = int(min(top_left[1], bottom_left[1]))
    max_col = int(max(top_right[1], bottom_right[1]))

    return im[min_row:max_row + 1, min_col:max_col + 1,:]

def preprocessImages():
    for image in range(len(images)):
        im = imread(basedir + images[image] + '.jpg')
        edges = getEdges(im, sigmas[image])
        # displayOriginalAndEdges(im, edges)
        ratio = getEdgesRatio(edges, images[image])
        ratios[images[image]] = ratio

def getEdges(im, s):
    # Run Canny Edge Detector on input image and return edges
    edges = feature.canny(rgb2gray(im), sigma=s)
    return edges

def displayOriginalAndEdges(im, edges):
    # Display the original image and the edges detected by the Canny Edge Detector
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].imshow(im)
    axs[0].set_title('Original Image')

    axs[1].imshow(edges, cmap='gray', interpolation=None)
    axs[1].set_title('Edges')
    plt.show()

def getEdgesRatio(edges, name):
    intensities = np.reshape(edges, (-1, 1))  # Reshape to 1-D
    counts, bins, bars = plt.hist(intensities, bins=2, edgecolor='black', linewidth=1.2)
    # plt.title("Histogram of " + name + " Image")
    # plt.show()
    return counts[1]/(counts[0] + counts[1])

def findMatch(im, r):
    for image, ratio in ratios.iteritems():
        percentDifference = abs(r - ratio) / (0.5*(r + ratio))
        if percentDifference <= threshold:
            results.append(image)
            results_indicies.append(images.index(image))
    results_indicies.sort()
    # print 'results: ', results

def displayMatches():
    print "Success" if inputImage in results else "Failure"
    print len(results)
    print results_indicies

    # i = len(results) + 1
    # plt.suptitle("Matches for " + inputImage)
    # plt.subplot(2, len(results), 1)
    # plt.title("Input Image - " + inputImage)
    # plt.imshow(imread('../images/' + inputImage + '.jpg'))
    #
    # for image in results:
    #     plt.subplot(2, len(results), i)
    #     plt.title("Output Image - " + image)
    #     plt.imshow(imread('../images/' + image + '.jpg'))
    #     i = i+1
    # plt.show()


def getSubsetWithEdgeAnalysis(inputImage, realImage):
    im = imread('../queryImages/' + inputImage + '.jpg')
    corners = getCoordinatesFromImage(im)
    im = getCroppedImage(im, corners)
    plt.title('Cropped Input Image')
    plt.imshow(im)
    plt.show()

    im = imread(basedir + realImage + '.jpg')
    preprocessImages()

    edges = getEdges(im, inputSigma)
    ratio = getEdgesRatio(edges, inputImage)

    findMatch(im, ratio)
    displayMatches()
    return results_indicies


if __name__ == '__main__':
    inputImage = 'mona-lisa'
    realImage = 'mona-lisa'
    results_indicies = getSubsetWithEdgeAnalysis(inputImage, realImage)

    # im = imread('../queryImages/' + inputImage + '.jpg')
    # corners = getCoordinatesFromImage(im)
    # im = getCroppedImage(im, corners)
    # plt.title('Cropped Input Image')
    # plt.imshow(im)
    # plt.show()
    #
    # im = imread(basedir + inputImage + '.jpg')
    # preprocessImages()
    #
    # edges = getEdges(im, inputSigma)
    # ratio = getEdgesRatio(edges, inputImage)
    #
    # findMatch(im, ratio)
    # displayMatches()
