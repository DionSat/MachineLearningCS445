"""
Author: Dion Satcher
Date: 3/13/20
Final Assignment
CS445
Student ID: 911832609
"""
import pandas as pd
import numpy as np
import math
import sklearn.svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

data = pd.read_csv('spambase.data', header=None, index_col=57)

# split data into train, test and train labels and test label
X_test, X_train, y_test, y_train = sklearn.model_selection.train_test_split(data, data.index.values,
                                                                            stratify=data.index.values,
                                                                            test_size=0.5)
# convert dataframe to numpy array
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# calculate mean of data columns
def calculate_mean(class_data):
    class_mean = np.mean(class_data, axis=0)
    return class_mean


# calculate standard deviation of columns
def calculate_std(class_data, epsilon):
    class_std = np.std(class_data, axis=0)
    class_std += epsilon
    return class_std

# calculate variance of the columns
def calculate_variance(class_std):
    class_var = np.square(class_std)
    return class_var

# class for gaussian naive bayes calculation
class GaussianNaiveBayes(object):

    def __init__(self):
        self.labels = []                        # list to the hold the unique labels
        self.means = {}                         # dictionary of the means of each labels columns
        self.stddevs = {}                       # dictionary of the standard deviation of each labels columns
        self.variance = {}                      # dictionary of the variance of each labels columns
        self.probability = {}                   # dictionary of the class label probabilities
        self.correctly_classified = 0           # the number of correctly classified
        self.epsilon = 0.0001
        self.pred_list = []

    # a function to train the model and calculate the means, standard deviations, variances and class probabilities
    def train(self, train_data, train_label):
        total_rows = train_data.shape[0] - 1                                    # total number of rows
        self.labels = np.unique(train_label)
        temp_train = train_data.tolist()
        for label in self.labels:
            class_data = []
            for i in range(len(train_label)):                                   # get data of label
                if train_label[i] == label:
                    class_data.append(temp_train[i])
            class_data = np.array(class_data)
            mean = calculate_mean(class_data)                                                       # calculate the means
            class_std = calculate_std(class_data, self.epsilon)                                     # calculate the standard deviations
            class_variance = calculate_variance(class_std)                                          # calculate the variance
            if label in self.means:
                self.means[label].append(mean)
                self.variance[label].append(class_variance)
                self.stddevs[label].append(class_std)
            else:
                self.means[label] = [mean]
                self.variance[label] = [class_variance]
                self.stddevs[label] = [class_std]
            for dimension in range(0, class_data.shape[1] - 1):
                print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (
                    label, dimension + 1, mean[dimension], class_std[dimension]))
        self.probability = self.probability_of_classifiers(total_rows, train_data, train_label)                 # calculate the probability of classifiers

    # calculate the probability of the classifiers
    def probability_of_classifiers(self, total_rows, data, data_labels):
        probability_dic = {}
        temp_train = data.tolist()
        for labels in self.labels:
            class_data = []
            for i in range(len(data)):  # get data of label
                if data_labels[i] == labels:
                    class_data.append(temp_train[i])
            probability = (len(class_data)) / float(total_rows)
            probability_dic[labels] = probability
        return probability_dic

    # a function to classify the data in the testing_data
    def test(self, testing_data, testing_label):
        total_rows = testing_data.shape[0] - 1
        row_id = 1
        for i in range(len(testing_data)):
            self.classify(testing_data[i], row_id, testing_label[i])
            row_id += 1
        self.display_accuracy(total_rows, testing_label, self.pred_list)
        self.print_confustion_matrix(testing_label, self.pred_list)
        print('Precision: ', precision_score(testing_label, self.pred_list))
        print('Recall: ', recall_score(testing_label, self.pred_list))

    def print_confustion_matrix(self, actual, predicted):
        print(confusion_matrix(actual, predicted))

    # a function to display hte accuracy
    def display_accuracy(self, total_rows, actual, pred):
        print("classification accuracy=%6.4lf " % accuracy_score(actual, pred))

    # a function to classify the row given by the data
    def classify(self, data_row, row_id, row_label):
        self.prob = {}
        for label in self.labels:
            for column in range(0, len(data_row) - 1):
                normal_result = self.calculuate_normal_dist(data_row[column], self.means[label][0][column],
                                                            self.variance[label][0][column],
                                                            self.stddevs[label][0][column])
                if label in self.prob:
                    self.prob[label] *= normal_result
                else:
                    self.prob[label] = normal_result
        for label in self.prob:
            self.prob[label] = np.log(self.prob[label] * self.probability[label])
        best_probability = self.prob[0]
        best_label = 0
        num_best = 0
        for label in self.labels:
            if self.prob[label] > best_probability:
                best_probability = self.prob[label]
                best_label = label
                num_best = 0
            if self.prob[label] == best_probability:
                num_best += 1

        accuracy = 0
        if num_best > 1 and row_label != best_label:
            accuracy = self.prob[self.rargmax(self.prob, best_probability)]
        if num_best > 1 and row_label == best_label:
            accuracy = 1 / num_best
        if best_label == row_label:
            accuracy = 1
        self.pred_list.append(best_label)
        print("ID = %5d, predicted = %3d, probability = %.6f, true=%3d, accuracy=%4.2f" % (
            row_id, best_label, best_probability, row_label, accuracy))
        return best_label

    # a function to choose a random value in a tie
    def rargmax(self, vector, best_prob):
        """ Argmax that chooses randomly among eligible maximum indices. """
        choice = []
        for x in range(len(vector)):
            if vector[x] == best_prob:
                choice.append(x)
        m = np.random.choice(choice, 1, replace=False)
        return m[0]

    # calculate the normal distribution or the gaussian
    def calculuate_normal_dist(self, value, mean, variance, std):
        denom = math.sqrt(2 * math.pi) * std
        power = ((-1) * math.pow((value - mean), 2)) / float((2 * variance))
        numerator = math.pow(math.e, power)
        normal = numerator / float(denom)

        return normal


gnb = GaussianNaiveBayes()
gnb.train(X_train, y_train)
gnb.test(X_test, y_test)
