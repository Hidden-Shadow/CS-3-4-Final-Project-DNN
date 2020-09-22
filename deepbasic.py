
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
# from main import *

def sigmoid(Z):
# Sigmoid Function    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache


def relu(Z):
# RELU function
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    dA - Post-activation gradient 
    cache - Where 'Z' is stored for computing backpropagation 
    
    returns dZ - Cost gradient w/ respect to Z
    """
    
    
    Z = cache
    dZ = np.array(dA, copy=True) # Just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    # Same arguments as above
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features (images)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels (cat / not cat)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features (images)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels (cat / not cat)
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    customimg = Image.open("images/realcat.jpg")
    #customimg = Image.open("images/plane.jpg") # uncomment these to switch which image to load.
    #customimg = Image.open("images/road.jpg")
    customimg_array = np.array(customimg)
    customimg_array = customimg_array[np.newaxis,...]
    print(customimg_array.shape)
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes, customimg, customimg_array


def initialize_parameters_deep(layer_dims): # Initializing parameters for DEEP
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # Returns the number of layers
    
    for l in range(1, L): # Loops throuugh the layers filling with random numbers 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) # Weights
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) #Biases
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

"""
def initialize_parameters(inputsize, hiddensize, outputsize): # Initializes Weights and Biases with random numbers for backpropagation.
    
    np.random.seed(1)
    
    W1 = np.random.randn(hiddensize, inputsize)*0.01
    b1 = np.zeros((hiddensize, 1))
    W2 = np.random.randn(outputsize, hiddensize)*0.01
    b2 = np.zeros((outputsize, 1))
    
    assert(W1.shape == (hiddensize, inputsize))
    assert(b1.shape == (hiddensize, 1))
    assert(W2.shape == (outputsize, hiddensize))
    assert(b2.shape == (outputsize, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters # Outputs parameter array
"""

def linear_forward(A, W, b):
    """
    A - Activations from prev layer
    W - Weight matrix
    b - Bias vector
    """
    
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    A_prev - Activations from prev layer
    W - Weight matrix
    b - Bias vector
    activation - type of filtering -- Sigmoid / RELU
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache # Returns post-activation value
    
    
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    X - Data
    parameters - Randomized W and b values from initialize_parameters_deep
    
    Outputs:
    AL - Last post-activation value (final result)
    caches - List of caches, kinda like a notebook?
    """
    
    caches = []
    A = X
    L = len(parameters) // 2 #of layers in neural network
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters ['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches
    
    
def compute_cost(AL, Y):
    
    m = Y.shape[1]
    
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost
        
    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1. / m) * np.dot(dZ, cache[0].T)  
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches): # Calculates backpropogation values
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
                                                                                                  
    for l in reversed(range(L-1)):                                                                                                 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate): #Changes parameters based on backpropagation values
    
    L = len(parameters) // 2 # number of layers in the neural network
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
        
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X - data set of examples you would like to label
    parameters - parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def predict_custom_image(X, parameters):
    #m = X.shape[1]
    #n = len(parameters)
    #p = np.zeros((1,m))

    probas, caches = L_model_forward(X, parameters)
    #print(probas.shape)
    catprobability = round(probas[0,0],2)*100
    print("Cat probability: " + str(catprobability) + '%')
    
    if catprobability > 50:
        print("This is most likely a cat.")
    else:
        print("This probably isn't a cat.")

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

