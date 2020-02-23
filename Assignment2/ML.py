"""
Author: Dion Satcher
Date: 2/11/20
Assignment 2
CS445
Student ID: 911832609
"""
import math
from scipy.special import expit
import numpy as np
from Assignment2 import loader
import itertools
import csv
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

train_data, test_data = loader.load_data_wrapper()

#====Experiment 2====
train_data = train_data[0:15000, :]

class NeuralNetwork(object):

    def __init__(self):
        self.input_size = 785                               #input size
        self.output_size = 10                               #output size
        self.hidden_size = 100                              #hidden layer size
        self.bias = 1                                       #bias for the data
        self.lr = 0.1                                       #the learning rate for the training
        self.accuracy = 0                                   #accuracy of the network after training
        self.train_acc = []                                 #list of training accuracy
        self.test_acc = []                                  #list of testing accuracy
        self.epoch_list = []                                #list of epochs for graphing
        y = []
        z = []

        "Initialize Neural Network Model"
        self.W1 = np.random.uniform(high=0.5, low= -0.5, size=(self.input_size, self.hidden_size))          #(20x785) weight matrix from input to hidden layer
        self.W2 = np.random.uniform(high=0.5, low=-0.5, size=(self.hidden_size + 1, self.output_size))      #(20x10) weight matrix from hidden to ouput layer

        # matrix to store the activation of hidden layer
        self.hl_input = np.zeros((1, self.hidden_size + 1))
        self.hl_input[0, 0] = 1

    """
    Turn the target number into an array
    """
    def target_result(self, j):
        e = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        e[j] = 0.9
        return e

    """
    Tune the input first and preprocess
    """
    def sampleAndTune(self, input, index):
        x = input[index] / 255      # scale the input
        x[0] = self.bias            # set bias at 0 position
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
    def feedforward(self, pred_list, input):
        #input to hidden
        out_hl = np.dot(input, self.W1)
        #sigmoid activation of hidden
        sig_hl = expit(out_hl)
        #add bias to output
        self.hl_input[0, 1:] = sig_hl
        #hidden layer to output
        out_ol = np.dot(self.hl_input, self.W2)
        #sigmoid activation of output
        output = expit(out_ol)
        #get the prediction
        pred_list.append(self.target_result(np.argmax(output)))
        return output, sig_hl

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
    def learn_algorithm(self, X, training_flag, epoch):
        con_outlist = []                                                                    #ouput list for the confusion matrix
        con_predlist = []                                                                   #prediction list for the confusion matrix
        actual_value = []                                                                   #target value list
        pred_list = []                                                                      #prediction list
        np.random.shuffle(X)                                                                #shuffle data
        for i in range(0, X.shape[0]):                                                      #go though the data set
            con_outlist.append(X[i, 0].astype(int))                                         #add the target to the output list for the confusion matrix
            target = self.target_result(X[i, 0].astype(int))                                #turn the 0th position target number into an array
            x = self.sampleAndTune(X, i)                                                    #sample input and tune it                                                              #the prediction list to hold the value that hasnt had hte activation function applied to it
            actual_value.append(target)
            output_o, output_h = self.feedforward(pred_list, x)                             # get the prediction of the output
            con_predlist.append(np.argmax(output_o))

            if (pred_list[i] != target):                                                    #check if the prediction was right and matched the target
                self.back_prop(training_flag, target, output_o, output_h, x, epoch)         #if the prediction was wrong backproprogate to calculate loss and update weights

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
    def back_prop(self, flag, target, output_o, output_h, input, epoch):
        if (flag == 1 and epoch > 0):
    #=====Calculate Error Term=====
            #error for output layer
            error_o = output_o*(1 - output_o)*(target - output_o)
            #error for hidden layer
            sum = np.dot(error_o, np.transpose(self.W2[1:, :]))
            error_h = output_h*(1 - output_h)*sum
            #=====Update Weights=====
            #Update weights from hidden to output
            delta_W2 = self.lr * error_o * np.transpose(self.hl_input)
            self.W2 += delta_W2

            #Update weights from input to hidden
            delta_W1 = self.lr * error_h * np.transpose(input)
            self.W1 += delta_W1


    #activation function
    def activation(self, X):
        i = 0;
        for x in X:
            if(X[i] > 0):
                X[i] = 1
            else:
                X[i] = 0
            i += 1

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #function to the plot the training from the .csv files
    def plot_acc(self):
        plt.figure(figsize=(15,5))
        plt.plot(NN.epoch_list, NN.train_acc, label="Training Set")
        plt.plot(NN.epoch_list, NN.test_acc, label="Testing Set")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('For Learning rate ' + str(self.lr))
        plt.legend()
        plt.show()

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("W1.txt", self.W1, fmt="%s")

NN = NeuralNetwork()
for i in range(50):    #train the NN 50 times
    print("Epoch # " + str(i) + "\n")
    NN.learn_algorithm(train_data, 1, i)
    train_accuracy = NN.accuracy
    print("The training accuracy of the given Epoch is: " + str(train_accuracy))
    NN.train_acc.append(train_accuracy)
    NN.learn_algorithm(test_data, 0, i)
    test_accuracy = NN.accuracy
    print("The testing accuracy of the given Epoch is: " + str(test_accuracy))
    NN.test_acc.append(test_accuracy)
    NN.epoch_list.append(i)

NN.plot_acc()