import numpy as np
import random as r
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys

number_clusters = 10
number_classes = 10
number_of_trials = 5


def load():
    train_labels = []
    train_features = []
    test_labels = []
    test_features = []

    training_data = pd.read_csv('./optdigits/optdigits.train', delimiter=',')
    test_data = pd.read_csv('./optdigits/optdigits.test', delimiter=',')
    training_data = pd.DataFrame(training_data).to_numpy()
    test_data = pd.DataFrame(test_data).to_numpy()

    """train_labels = np.unique(training_data[:, training_data.shape[1] - 1])
    train_labels = np.sort(train_labels)"""
    train_labels = training_data[:, -1]
    # train_features = np.delete(training_data, training_data.shape[1] - 1, 1)
    train_features = training_data
    """test_labels = np.unique(training_data[:, training_data.shape[1] - 1])
    test_labels = np.sort(test_labels)"""
    test_labels = test_data[:, -1]
    # test_features = np.delete(training_data, training_data.shape[1] - 1, 1)
    test_features = test_data
    return train_features, train_labels, test_features, test_labels


def main():
    train_features, train_labels, test_features, test_labels = load()
    best_run = kmeans(train_features, train_labels, number_clusters)
    test_clusters = partitionData(test_features, best_run[0])
    i, j = test_features.shape
    print("Test: Sum square sep: ", sumSquaredSeparation(test_clusters))
    print("Test: Entropy: ", meanEntropy(test_clusters, number_clusters, i))
    print("Test: Sum squared Error:", sumSquaredError(best_run[0], test_clusters))


def create_confusion_matrix(features, clusters):
    confusion_matrix = [[0 for i in range(number_classes)] for i in
                        range(number_classes)]
    correct_classification = 0
    for i in range(len(clusters)):
        for x in clusters[i]:
            if x[-1] == i:
                correct_classification += 1


def distance(feature, centerList):
    """Take a feature vector and a list of center vectors and calculate the euclidean distance
       from the feature vector to each center vector. Return a list of distances."""

    distances = []
    attributes = feature[:-1]  # remove the label
    for each in range(0, len(centerList)):
        center = centerList.pop(each)
        summation = 0
        for i in range(0, len(attributes)):
            if len(center) == 0:
                break
            summation += np.square(center[i] - attributes[i])
        distance = np.sqrt(summation)
        distances.append(distance)
        centerList.insert(each, center)
    return distances


def distancePoint(point, center):
    square_sums = 0.0
    for point_i, center_i in zip(point, center):
        square_sums += (point_i - center_i) ** 2
    return np.sqrt(square_sums)


def closest_center(point, centers):
    distances = list()
    for center in centers:
        distances.append(distance(point, center))
    dist_array = np.array(distances)

    first_min_distance = dist_array.argmin()

    min_distances = list()
    for i in range(len(distances)):
        if distances[i] - distances[first_min_distance] < 10 ** -10:
            min_distances.append(i)
    return np.random.choice(min_distances)


def partitionData(matrix, centers):
    i, j = matrix.shape
    numCenters = len(centers)
    clusters = [[] for i in range(numCenters)]

    for row in range(0, i):
        distanceList = []
        distanceList = distance(matrix[row], centers)
        centerIndex = np.argmin(distanceList)
        clusters[centerIndex].append(list(matrix[row]))
    return clusters


def check_if_centers_updated(old_centers, centers, epoch):
    if epoch > 0:
        old_centers = np.array(old_centers, dtype=float)
        centers = np.array(centers, dtype=float)
        old_centers = old_centers.tolist()
        centers = centers.tolist()
        if np.allclose(old_centers, centers, rtol=0.1, atol=0.01):
            return True
        else:
            return False
    else:
        return False


def kmeans(data, data_label, k):
    trial_data = dict()
    sseRun = []
    sssRun = []
    centerList = []
    entropyRun = []
    i, j = data.shape
    for trial in range(number_of_trials):
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
            print('Epoch: %d' % epoch)
            clustering = partitionData(data, centers)
            sss = sumSquaredSeparation(centers)
            mec = meanEntropy(clustering, k, i)
            sse = sumSquaredError(centers, clustering)
            entropyRun.append(mec)
            sseRun.append(sse)
            sssRun.append(sss)
            centerList.append(centers)
            old_center = centers
            centers = updateCentroids(clustering, data, centers, epoch)
            change = check_if_centers_updated(old_center, centers, epoch)
            epoch += 1
        print('\nFinal cluster centers for trial %d set.' % trial)
        for print_l in centers:
            print(print_l)
        trial_data[trial] = [centers, sse, sss, mec]
    print('All %d trials have finished' % number_of_trials)
    for i in range(number_of_trials):
        print('Trial: %d' % i)
        print('Mean Square Error: %f' % trial_data[i][1])
        print('Mean Square Separation: %f' % trial_data[i][2])
        print('Mean Entropy: %f' % trial_data[i][3])
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
    newVector = []
    newCenters = []
    for i in range(0, len(clusters)):
        if len(clusters[i]) > 0:
            for j in clusters[i]:
                del j[-1]
            newCenters.append(list(np.mean(clusters[i], axis=0, dtype=np.int)))
        else:
            """for i in range(0, 64):
                newVector.append(r.uniform(0, 17))
            newCenters.append(newVector)"""
            newCenters.append(centers[i])

    return newCenters


def tally(cluster):
    """count the number of y's in an tuple (x,y). Return the number of y's"""

    counter = []
    if len(cluster) > 0:
        value_compare = []
        for x in cluster:
            value_compare.append(x[-1])
        counter = np.bincount(value_compare)
    return counter


def meanEntropy(clusters, k, dataSize):
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
        count = tally(cluster)
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


def sumSquaredError(centers, clusters):
    err = 0
    count = 0
    inner_term = 0
    for point in range(0, len(centers)):
        feature = clusters[point]
        for i in feature:
            inner_term = np.square(np.asarray(i[:-1]) - np.asarray(centers[point])).sum(axis=0)
            err += np.sqrt(inner_term)
            inner_term = 0
            count += 1
    err = err / count

    return err


def sumSquaredSeparation(clusters):
    """Take a list of lists and find the sum squared separation(SSS). Return the SSS."""
    summation = 0
    err = 0
    for each in range(0, len(clusters)):
        cluster = clusters.pop(each)
        for i in clusters:
            summation = np.square(np.asarray(cluster) - np.asarray(i)).sum(axis=0)
            err += np.sqrt(summation)
            summation = 0
        clusters.insert(each, cluster)
    err = err / (number_clusters * (number_clusters - 1) / 2)
    return err


def printConfusion(self, predictions, targets, epoch):
    print("The confusion matrix for the test data at epoch " + str(epoch))
    print(confusion_matrix(targets, predictions))


if __name__ == "__main__":
    main()
