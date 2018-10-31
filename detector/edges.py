import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage import feature
from skimage.color import rgb2gray

# mystery-history - https://www.wikiart.org/en/myron-stout/untitled-may-20-1950
# tomb (Impressionism) - https://www.wikiart.org/en/owen-jones/tomb-near-cairo-1833
# spectrum-colors (Hard Edge Paintings) - https://www.wikiart.org/en/ellsworth-kelly/spectrum-colors-arranged-by-chance-ii-1951
# abstract - https://www.wikiart.org/en/myron-stout/untitled-1948
# synchromy-in-orange - https://www.wikiart.org/en/morgan-russell/synchromy-in-orange-to-form-1914
# triangle - https://www.wikiart.org/en/ellsworth-kelly/triangle-form-1951
# two-edges - https://www.wikiart.org/en/barnett-newman/two-edges-1948
# three-rocks - https://www.wikiart.org/en/john-ferren/three-rocks-1949
# john-ferren - https://www.wikiart.org/en/john-ferren/untitled-1952-1

# Good Results:
# inputImage = 'john-ferren'
# inputSigma = 0.55
#
# inputImage = 'tomb'
# inputSigma = 1.7
#
#
# Not so great Results:
# inputImage = 'abstract'
# inputSigma = 1.5
#
# inputImage = 'synchromy-in-orange'
# inputSigma = 1.2

inputImage = 'john-ferren'
inputSigma = 0.55
ratios = {}
images = ['mystery-history', 'tomb', 'spectrum-colors', 'abstract', 'synchromy-in-orange',
            'triangle', 'two-edges', 'three-rocks', 'john-ferren']
sigmas = [3.5, 1.8, 1, 2, 1.3, 2, 3, 2, 0.5]
threshold = 0.1
results = []

def preprocessImages():
    for image in range(len(images)):
        im = imread('../images/' + images[image] + '.jpg')
        edges = getEdges(im, sigmas[image])
        # displayOriginalAndEdges(im, edges)
        ratio = getEdgesRatio(edges, images[image])
        ratios[images[image]] = ratio
    # print 'ratios: ', ratios

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
    # Why do values range from 0 to 1 and not 0 to 255
    intensities = np.reshape(edges, (-1, 1))  # Reshape to 1-D
    counts, bins, bars = plt.hist(intensities, bins=2, edgecolor='black', linewidth=1.2)
    plt.title("Histogram of " + name + " Image")
    # plt.show()
    return counts[1]/(counts[0] + counts[1])

def findMatch(im, r):
    for image, ratio in ratios.iteritems():
        percentDifference = abs(r - ratio) / (0.5*(r + ratio))
        if percentDifference <= threshold:
            results.append(image)
    print 'results: ', results

def displayMatches():
    i = len(results) + 1
    plt.suptitle("Matches for " + inputImage)
    plt.subplot(2, len(results), 1)
    plt.title("Input Image - " + inputImage)
    plt.imshow(imread('../images/' + inputImage + '.jpg'))

    for image in results:
        plt.subplot(2, len(results), i)
        plt.title("Output Image - " + image)
        plt.imshow(imread('../images/' + image + '.jpg'))
        i = i+1
    plt.show()



if __name__ == '__main__':
    preprocessImages()

    im = imread('../images/' + inputImage + '.jpg')
    # plt.imshow(im)
    # plt.title("Input Image " + inputImage)
    # plt.show()
    edges = getEdges(im, inputSigma)
    ratio = getEdgesRatio(edges, inputImage)
    print 'r: ', ratio

    findMatch(im, ratio)
    displayMatches()
