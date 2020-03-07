"""
Dion Satcher
3/6/20
CS445
Assignment 4
Student ID: 911832609
"""

import numpy as np
import random as r
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image
from matplotlib import pyplot as plt

number_clusters = 10                        # number of cluster
number_classes = 10                         # number of classes
number_of_trials = 5                        # number of trials


def load():
    training_data = pd.read_csv('./optdigits/optdigits.train', delimiter=',')           # read training data using panda
    test_data = pd.read_csv('./optdigits/optdigits.test', delimiter=',')                # read test data using panda
    training_data = pd.DataFrame(training_data).to_numpy()                              # convert to numpy array
    test_data = pd.DataFrame(test_data).to_numpy()                                      # convert to numpy array

    train_labels = training_data[:, -1]                                                 # get column of labels from training data
    train_features = training_data
    test_labels = test_data[:, -1]                                                      # get column of labels from test data
    test_features = test_data
    return train_features, train_labels, test_features, test_labels


def main():
    train_features, train_labels, test_features, test_labels = load()                                   # load the data
    # run the k-means algorithm on the training data and return the best run
    # ========= Experiment 1 ================
    number_clusters = 10
    best_run = kmeans(train_features, train_labels, number_clusters)                                    # run the k-means algorithm
    test_clusters, prediction_list = partitionData(test_features, best_run[0])                          # partition test data into closest centers
    i, j = test_features.shape                                                                          # get the row and column numbers of the data
    print()
    # calculate the sum square separation, mean square error and entropy of the test set
    print('Using the best yield centers for the test data')
    print("Test: Sum square sep: ", sumSquaredSeparation(best_run[0], test_features))
    print("Test: Entropy: ", meanEntropy(test_clusters, number_clusters, i, test_features))
    print("Test: Sum squared Error:", sumSquaredError(best_run[0], test_clusters, test_features))
    most_frequent = accuracy_metric(test_clusters, best_run[0], i, test_features)                       #classify clusters
    # classify clusters by the most frequent value in the cluster
    printConfusion(most_frequent, test_clusters, test_features)
    # print the grayscale image of the best centers
    print('Centers Labels: ', most_frequent)
    for i in range(number_clusters):
        draw_center_as_bitmap(best_run[0][i])
    print()

    # =========== Experiment 2 ===============
    number_clusters = 30
    best_run = kmeans(train_features, train_labels, number_clusters)  # run the k-means algorithm
    test_clusters, prediction_list = partitionData(test_features,
                                                   best_run[0])  # partition test data into closest centers
    i, j = test_features.shape  # get the row and column numbers of the data
    print()
    # calculate the sum square separation, mean square error and entropy of the test set
    print('Using the best yield centers for the test data')
    print("Test: Sum square sep: ", sumSquaredSeparation(best_run[0], test_features))
    print("Test: Entropy: ", meanEntropy(test_clusters, number_clusters, i, test_features))
    print("Test: Sum squared Error:", sumSquaredError(best_run[0], test_clusters, test_features))
    most_frequent = accuracy_metric(test_clusters, best_run[0], i, test_features)  # classify clusters
    # classify clusters by the most frequent value in the cluster
    printConfusion(most_frequent, test_clusters, test_features)
    # print the grayscale image of the best centers
    print('Centers Labels: ', most_frequent)
    for i in range(number_clusters):
        draw_center_as_bitmap(best_run[0][i])
    print()


def draw_center_as_bitmap(center):
    #draw the center as a bitmap
    center_2d = np.array(center).reshape(8, 8)          # reshape into 8x8 matrix
    center_2d = center_2d * 16                          # turn values into grayscale intesity
    plt.gray()
    plt.imshow(center_2d)
    plt.show()


def distance(feature, centerList):
    """Take a feature vector and a list of center vectors and calculate the euclidean distance
       from the feature vector to each center vector. Return a list of distances."""

    distances = []                                                          # list of distances
    attributes = feature[:-1]                                               # remove the label from features
    attributes = list(attributes)                                           # turn into a list
    for each in range(0, len(centerList)):                                  # for each center in the center list
        center = centerList.pop(each)                                       # take a center
        summation = 0
        summation = np.square(np.array(center) - np.array(attributes))
        summation = summation.sum()
        distance = np.sqrt(summation)
        distances.append(distance)
        centerList.insert(each, center)
    return distances


def partitionData(matrix, centers):
    i, j = matrix.shape                                     # get row and column numbers
    numCenters = len(centers)                               # get how many centers there are
    clusters = [[] for i in range(numCenters)]              # create empty clusters to fill
    pred_list = []

    for row in range(0, i):
        # return a list of distances from the data row(point) to centers
        distanceList = distance(matrix[row], centers)       # get distance from data row to centers
        centerIndex = np.argmin(distanceList)               # get index with smallest distance
        clusters[centerIndex].append(row)
        pred_list.append(centerIndex)
    return clusters, pred_list


def check_if_centers_updated(old_centers, centers, epoch):
    """
    Check if the center has changed since the last iteration
    :param old_centers: old center from last iteration
    :param centers:     current centers
    :param epoch:       the current epoch
    :return: true if no change or false
    """
    # if its the first epoch there is no point in checking
    if epoch > 0:
        # convert centers to numpy arrays then floats in order to do calculations(comparison)
        old_centers = np.array(old_centers, dtype=float)
        centers = np.array(centers, dtype=float)
        old_centers = old_centers.tolist()
        centers = centers.tolist()
        if np.allclose(old_centers, centers, rtol=0.0001, atol=0.00001):
            return True
        else:
            return False
    else:
        return False


def CountFrequency(my_list, data):
    """
    Count frequency in cluster
    :param my_list: cluster
    :param data:    data to reference
    :return:        list with indices corresponding to frequency
    """
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if data[item, -1] in freq:
            freq[data[item, -1]] += 1
        else:
            freq[data[item, -1]] = 1

    return freq


def accuracy_metric(clusters, centers, data_num, data):
    mostFreqPerCluster = []
    accuracy = 0
    for each in range(0, len(clusters)):
        count = 0
        classes = []
        freq = []
        cluster = clusters.pop(each)
        for i in cluster:
            classes.append(list(data[i, :-1]))
        freq = CountFrequency(cluster, data)
        if freq:
            frequent = max(freq, key=freq.get)
            accuracy += freq[frequent]
            mostFreqPerCluster.append(frequent)
        else:
            mostFreqPerCluster.append(None)
        clusters.insert(each, cluster)
    print("Accuracy:", (accuracy / data_num))

    return mostFreqPerCluster


def kmeans(data, data_label, k):
    temp_data_label = list(data_label)
    trial_data = dict()
    sseRun = []
    sssRun = []
    centerList = []
    entropyRun = []
    i, j = data.shape
    for trial in range(number_of_trials):
        pred_list = []
        epoch = 0
        print('Beginning trial #%d...' % trial)
        # Initialize centers for each cluster randomly
        print('Initializing random centers for %d clusters' % number_clusters)
        centers = centroid(k)
        # Repeat until the center does not move or oscillate
        print('Working on finding cluster centers')
        change = False

        # Stop iterating K-Means when all cluster centers stop changing
        while change is False:
            pred_list = []
            print('Epoch: %d' % epoch)
            clustering, pred_list = partitionData(data, centers)
            sss = sumSquaredSeparation(centers, data)
            mec = meanEntropy(clustering, k, i, data)
            sse = sumSquaredError(centers, clustering, data)
            entropyRun.append(mec)
            sseRun.append(sse)
            sssRun.append(sss)
            centerList.append(centers)
            old_center = centers
            centers = updateCentroids(clustering, data, centers, epoch)
            change = check_if_centers_updated(old_center, centers, epoch)
            epoch += 1
            print('Mean Square Error for epoch is: %f' % sse)
            print('Mean Square Separation for epoch is: %f' % sss)
            print('Mean Entropy for epoch is: %f' % mec)
            print()
        print()
        print('=============================')
        print('\nFinal cluster centers for trial %d set.' % trial)
        for print_l in centers:
            print(print_l)
        trial_data[trial] = [centers, sse, sss, mec, pred_list, clustering]
    print('All %d trials have finished' % number_of_trials)
    print()
    for tr in range(number_of_trials):
        print('Trial: %d' % tr)
        print('Mean Square Error: %f' % trial_data[tr][1])
        print('Mean Square Separation: %f' % trial_data[tr][2])
        print('Mean Entropy: %f' % trial_data[tr][3])
        accuracy_metric(trial_data[tr][5], centers, i, data)
    smallest_sse = 0
    for trial in range(1, len(trial_data)):
        if trial_data[trial][1] < trial_data[smallest_sse][1]:
            smallest_sse = trial
    print('The trial that yielded the smallest mean square error is: %d' % smallest_sse)

    return trial_data[smallest_sse]


def updateCentroids(clusters, data, centers, epoch):
    """Take a list of clusters and update the centers for each cluster. Each cluster contains a list of lists of
        feature vectors. Find the mean of each cluster, and assign the center to this value. Return the new
        center values"""
    summation = 0
    num_in_cluster = 0
    newVector = []
    newCenters = []
    for i in range(0, len(clusters)):
        if len(clusters[i]) > 0:
            for x in clusters[i]:
                summation += data[x, :-1]
                num_in_cluster += 1
            data_point = summation / num_in_cluster
            newCenters.append(list(data_point))
            num_in_cluster = 0
            summation = 0
        else:
            newCenters.append(centers[i])

    return newCenters


def tally(cluster, data):
    """count the number of y's in an tuple (x,y). Return the number of y's"""

    counter = []
    if len(cluster) > 0:
        value_compare = []
        for x in cluster:
            value_compare.append(data[x, -1])
        counter = np.bincount(value_compare)
    return counter


def meanEntropy(clusters, k, dataSize, data):
    """Take a list of list of lists of clusters, the number of cluster centers and the number of feature vectors
        and calculate the mean entropy of the cluster(MEC). Return the MEC"""

    entropy = 0
    log = 0
    coeff = []
    clusterSize = []
    for each in range(0, len(clusters)):
        cluster = clusters[each]
        clusterSize.append(len(cluster))
        clusterTotal = len(cluster)
        count = tally(cluster, data)
        if len(count) > 0:
            for label in range(len(count)):
                if count[label] != 0:
                    probability = count[label] / clusterTotal
                    log += probability * np.log2(probability)
        coeff.append(-log)
        log = 0

    for i in range(0, len(clusters)):
        entropy += ((clusterSize[i] / (dataSize)) * coeff[i])
    return entropy


def centroid(num_centers):
    """Take an integer and create that many lists containing 64 random integers.
        Return the list of lists of random integers"""
    centers = []
    center_list = []
    for each in range(0, num_centers):
        centers = []
        for each in range(0, 64):
            centers.append(int(r.uniform(0, 17)))
        center_list.append(centers)
    return center_list


def sumSquaredError(centers, clusters, data):
    err = 0
    count = 0
    inner_term = 0
    for point in range(0, len(centers)):
        feature = clusters[point]
        for i in feature:
            inner_term = np.square(np.asarray(data[i, :-1]) - np.asarray(centers[point])).sum(axis=0)
            err += np.sqrt(inner_term)
            inner_term = 0
            count += 1
    err = err / count

    return err


def sumSquaredSeparation(clusters, data):
    """Take a list of lists and find the sum squared separation(SSS). Return the SSS."""
    summation = 0
    err = 0
    for each in range(0, len(clusters)):
        cluster = clusters[each]
        for i in clusters:
            summation = np.square(np.asarray(cluster) - np.asarray(i)).sum(axis=0)
            err += np.sqrt(summation)
            summation = 0
    err = err / (number_clusters * (number_clusters - 1) / 2)
    return err


def printConfusion(cluster_label, clusters, data):
    # print confusion matrix
    pred_list = []
    actual = []
    # make prediction and actual list for confusion matrix
    for i in range(len(clusters)):
        for cluster in clusters[i]:
            pred_list.append(cluster_label[i])
            actual.append(data[cluster, -1])
    print('Confusion Matrix')
    print(confusion_matrix(actual, pred_list))


if __name__ == "__main__":
    main()
