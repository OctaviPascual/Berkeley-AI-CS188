# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import division
import numpy as np
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

        Without enhanced features, we get 95.9% accuracy on the training set,
        86.1% accuracy on the validation set and 87.0% on the test set.

        Enhanced feature 1: number of white regions that we can find in the image

        We distinguish between one, two and three white regions. However, an image
        could have no white region at all or more than three regions. If that is the case,
        we consider that no white region is the same as one region and more than
        three regions is the same as three regions. We perform that conversion because
        features must be encoded using binary features. The encoding is the following:
            - [1, 0, 0] -> at most one white region
            - [0, 1, 0] -> exactly two white regions
            - [0, 0, 1] -> at least three white regions

        With that feature alone, we get 96.7% accuracy on the training set,
        88.3% accuracy on the validation set and 89.4% on the test set.

        Enhanced feature 2: ratio of black pixels inside the digit bounding box

        We compute the number of black pixels of the image and the number of pixels
        of the bounding box (by definition, all black pixels are inside it).
        Then we use both values to compute the ratio of black pixels in the bounding box.
        (By doing that we minimize the impact of different scales in images).
        To encode those rations, we have to perform a discretization process.
        (I empirically came up to this encoding, seems the one which better works).
        The encoding is the following:
            - [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] -> ratio between 0 and 0.1
            - [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] -> ratio between 0.1 and 0.2
            - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> ratio between 0.2 and 0.3
            ...
            - [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] -> ratio between 0.8 and 0.9
            - [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] -> ratio between 0.9 and 1

        With that feature alone, we get 96.2% accuracy on the training set,
        86.0% accuracy on the validation set and 87.4% on the test set.

        Finally, with both features, we get 97.0% accuracy on the training set,
        88.7% accuracy on the validation set and 89.5% on the test set.
    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    regions = white_regions(datum)
    regions = min(regions, 3)
    regions = max(regions, 1)
    enhanced_feature_1 = np.zeros(3, dtype=int)
    enhanced_feature_1[regions - 1] = 1

    black_pixels_ratio = number_of_black_pixels(datum) / pixels_inside_bounding_box(datum)
    black_pixels_ratio = min(black_pixels_ratio, 0.99)
    enhanced_feature_2 = np.zeros(10, dtype=int)
    enhanced_feature_2[int(black_pixels_ratio*10)] = 1

    enhanced_features = np.concatenate((enhanced_feature_1, enhanced_feature_2))
    return np.concatenate((features, enhanced_features))

def white_regions(datum):
    visited = set()
    regions = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if (i, j) not in visited and datum[i][j] == 0:
                dfs(datum, i, j, visited)
                regions += 1
    return regions

def dfs(datum, i, j, visited):
    if 0 <= i < DIGIT_DATUM_HEIGHT and 0 <= j < DIGIT_DATUM_WIDTH and datum[i][j] == 0:
        visited.add((i, j))
        for x, y in neigbours(i, j):
            if (x, y) not in visited: dfs(datum, x, y, visited)

def neigbours(i, j):
    return [(i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)]

def pixels_inside_bounding_box(datum):
    min_i = DIGIT_DATUM_HEIGHT - 1
    max_i = 0
    min_j = DIGIT_DATUM_WIDTH - 1
    max_j = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if datum[i][j] > 0:
                min_i = min(i, min_i)
                max_i = max(i, max_i)
                min_j = min(j, min_j)
                max_j = max(j, max_j)
    return (max_j - min_j + 1) * (max_i - min_i + 1)

def number_of_black_pixels(datum):
    black_pixels = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if datum[i][j] > 0:
                black_pixels += 1
    return black_pixels

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    for i in range(len(validationPredictions)):
        prediction = validationPredictions[i]
        truth = valLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %d; truth is %d" % (prediction, truth)
            print "Image: "
            print_digit(valData[i,:])

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
