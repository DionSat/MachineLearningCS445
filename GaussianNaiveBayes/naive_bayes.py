import numpy as np
import math
import pandas as pd
import sys
import random as pr

training_data = pd.read_csv(sys.argv[1], delim_whitespace=True)
test_data = pd.read_csv(sys.argv[2], delim_whitespace=True)
training_data = pd.DataFrame(training_data).to_numpy()
test_data = pd.DataFrame(test_data).to_numpy()


def calculate_mean(class_data):
    class_mean = np.mean(class_data, axis=0)
    return class_mean


def calculate_std(class_data):
    class_std = np.std(class_data, axis=0)
    class_std = np.where(class_std == 0, 0.01, class_std)
    return class_std


def calculate_variance(class_std):
    class_var = np.square(class_std)
    #class_var = np.where(class_var == 0, 0.0001, class_var)
    return class_var


class GaussianNaiveBayes(object):

    def __init__(self):
        self.labels = []
        self.means = {}
        self.stddevs = {}
        self.variance = {}
        self.probability = {}
        self.correctly_classified = 0

    def train(self, train_data):
        self.labels = np.unique(train_data[:, train_data.shape[1] - 1])
        self.labels = np.sort(self.labels)
        total_rows = train_data.shape[0] - 1
        for label in self.labels:
            class_data = train_data[np.where(train_data[:, train_data.shape[1] - 1] == label)]
            #class_data = class_data[:, 0:class_data.shape[1] - 1]
            class_data = np.delete(class_data, train_data.shape[1] - 1, 1)
            mean = calculate_mean(class_data)
            class_std = calculate_std(class_data)
            class_variance = calculate_variance(class_std)
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
        self.probability = self.probability_of_classifiers(total_rows, train_data)

    def probability_of_classifiers(self, total_rows, data):
        probability_dic = {}
        for labels in self.labels:
            class_data = data[np.where(data[:, data.shape[1] - 1] == labels)]
            probability = (len(class_data)) / float(total_rows)
            probability_dic[labels] = probability
        return probability_dic

    def test(self, testing_data):
        total_rows = testing_data.shape[0] - 1
        row_id = 1
        for row in testing_data:
            self.classify(row, row_id)
            row_id += 1
        self.display_accuracy(total_rows)

    def display_accuracy(self, total_rows):
        print("classification accuracy=%6.4lf " % (self.correctly_classified / float(total_rows)))

    def classify(self, data_row, row_id):
        self.prob = {}
        for label in self.labels:
            product = 1.00
            for column in range(0, len(data_row) - 1):
                normal_result = self.calculuate_normal_dist(data_row[column], self.means[label][0][column],
                                                       self.variance[label][0][column], self.stddevs[label][0][column])
                if label in self.prob:
                    self.prob[label] *= normal_result
                else:
                    self.prob[label] = normal_result
        for label in self.prob:
            self.prob[label] = self.prob[label] * self.probability[label]
        denominator = sum(self.prob.values())
        best_probability = 0
        best_label = " "
        num_best = 0
        for label in self.labels:
            self.prob[label] /= (float(denominator))
            if self.prob[label] > best_probability:
                best_probability = self.prob[label]
                best_label = label
                num_best = 0
            if self.prob[label] == best_probability:
                num_best += 1

        accuracy = 0
        if num_best > 1 and data_row[-1] != best_label:
            accuracy = self.prob[self.rargmax(self.prob)]
        if num_best > 1 and data_row[-1] == best_label:
            accuracy = 1 / num_best
        print(data_row[-1])
        if best_label == data_row[-1]:
            self.correctly_classified += 1
            accuracy = 1
        print("ID = %5d, predicted = %3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            row_id, best_label, best_probability, data_row[-1], accuracy))
        return best_label

    def rargmax(self, vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        m = np.amax(vector)
        indices = np.nonzero(vector == m)
        return pr.choice(indices)

    def calculuate_normal_dist(self, value, mean, variance, std):
        if variance < 0.0001:
            variance = 0.0001
        denom = math.sqrt(2 * math.pi) * std
        power = ((-1) * math.pow((value - mean), 2)) / float((2 * variance))
        numerator = math.pow(math.e, power)
        normal = numerator / float(denom)

        return normal


gnb = GaussianNaiveBayes()
gnb.train(training_data)
gnb.test(test_data)
