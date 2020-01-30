"""
Author: Dion Satcher
Date: 1/26/20
Assignment 1
CS445
"""

import numpy as np
from PerceptronNetwork import loader
import csv
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

train_data, test_data = loader.load_data_wrapper()

class NeuralNetwork(object):

    def __init__(self):
        input_size = 785                            #input size
        output_size = 10                            #output size
        self.bias = 1                               #bias for the data
        self.lr = 0.001                             #the learning rate for the training
        self.accuracy = 0                           #accuracy of the network after training
        self.train_acc = []                         #list of training accuracy
        self.test_acc = []                          #list of testing accuracy
        y = []
        z = []

        "Initialize Neural Network Model"
        self.W1 = np.random.uniform(high=0.5, low= -0.5, size=(output_size, input_size))   #(10x785) weight matrix from input to ouput layer

    """
    Turn the target number into an array
    """
    def target_result(self, j):
        e = [0,0,0,0,0,0,0,0,0,0]
        e[j] = 1.0
        return e

    """
    Tune the input first and preprocess
    """
    def sampleAndTune(self, input, index):
        x = input[index] / 255  # scale the input
        x[0] = self.bias  # set bias at 0 position
        x = x.reshape(1, 785)
        return x


    """
    Function to print the confusion matrix
    """
    def printConfusion(self, predictions, targets, flag, epoch):
        if(flag == 0):
            print("The confusion matrix for the test data at epoch " + str(epoch))
            print(confusion_matrix(targets, predictions))

    """
    The function that will predict the number of the input using linear algebra and activation functions
    """
    def predict(self, output_list, pred_list, input):
        for j in range(10):                             #cycle through perceptron weights
            output = np.inner(input, self.W1[j, :])     #do inner product to get the 1x10 output of each perceptron
            out = np.copy(output)                       #copy the output so the output doesn't get modified
            pred_list.append(out)                       #add the output to the prediction list
            self.activation(output)                     #apply the activation function to the output
            output_list.append(output)                  #append the output to the output list

    def convertToInt(self, prediction, actual):
        x = 0
        value = 0;
        for i in prediction:
            if(i == 1):
                value = x
            x = x + 1

    """
    This is the main learning function that cycles through all of the 60000 inputs and tunes the weights to recognize the numbers
    """
    def perceptronLearn(self, X, training_flag, epoch):
        con_outlist = []                                                                #ouput list for the confusion matrix
        con_predlist = []                                                               #prediction list for the confusion matrix
        actual_value = []                                                               #target value list
        pred_list = []                                                                  #prediction list
        np.random.shuffle(X)                                                            #shuffle data
        for i in range(0, X.shape[0]):                                                  #go though the data set
            con_outlist.append(X[i, 0].astype(int))                                     #add the target to the output list for the confusion matrix
            target = self.target_result(X[i, 0].astype(int))                            #turn the 0th position target number into an array
            x = self.sampleAndTune(X, i)                                                #sample input and tune it
            aoutput_list = []                                                           #output list to hold the output after the activation funciton is applied
            p_list = []                                                                 #the prediction list to hold the value that hasnt had hte activation function applied to it
            actual_value.append(target)
            t = np.copy(target)
            self.predict(aoutput_list, p_list, x)                                       #get the prediction of the output
            prediction = np.array(p_list)
            con_pred = np.argmax(prediction)
            con_predlist.append(con_pred.astype(int))
            pred_list.append(self.target_result(np.argmax(prediction)))
            if (pred_list[i] != target):                                                #check if the prediction was right and matched the target
                self.tuneWeights(training_flag, target, aoutput_list, x, epoch)         #if the prediction was wrong then tune the weights

        self.accuracy_metric(actual_value, pred_list)
        self.printConfusion(con_predlist, con_outlist, training_flag, epoch)

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        self.accuracy = correct / float(len(actual)) * 100.0

    # Tune the weights
    def tuneWeights(self, flag, target, output_list, input, epoch):
        if (flag == 1 and epoch > 0):
            for i in range(10):
                self.W1[i, :] = self.W1[i, :] + (self.lr * (target[i] - output_list[i]) * input)

    #activation function
    def activation(self, X):
        i = 0;
        for x in X:
            if(X[i] > 0):
                X[i] = 1
            else:
                X[i] = 0
            i += 1

    #function to the plot the training from the .csv files
    def plot(self):
        x1, y1 = np.loadtxt("train_output0.001.csv", delimiter=',', unpack=True)
        x2, y2 = np.loadtxt("test_output0.001.csv", delimiter=',', unpack=True)
        plt.plot(x1, y1, label="Training Set")
        plt.plot(x2, y2, label="Testing Set")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%) ')
        plt.title('For Learning rate ' + str(self.lr))
        plt.legend()
        plt.show()

    def write_accuracy(self, accur_index, accur, input_ds):
        with open(input_ds, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([accur_index, accur])

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("W1.txt", self.W1, fmt="%s")

NN = NeuralNetwork()
for i in range(50):    #train the NN 50 times
    print("Epoch # " + str(i) + "\n")
    NN.perceptronLearn(train_data, 1, i)
    train_accuracy = NN.accuracy
    print("The training accuracy of the given Epoch is: " + str(train_accuracy))
    NN.train_acc.append(train_accuracy)
    NN.perceptronLearn(test_data, 0, i)
    test_accuracy = NN.accuracy
    print("The testing accuracy of the given Epoch is: " + str(test_accuracy))
    NN.test_acc.append(test_accuracy)
    NN.write_accuracy(i, train_accuracy, 'train_output' + str(NN.lr) + '.csv')
    NN.write_accuracy(i, test_accuracy, 'test_output' + str(NN.lr) + '.csv')

NN.plot()