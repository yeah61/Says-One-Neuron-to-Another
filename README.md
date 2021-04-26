# Says One Neuron To Another

  EECS738 (Spring 2021) assignment 3

  Author: Ye Wang

## Purpose

The purpose of this project is to understand the underlying mathematics of neural networks.Especially,to understand how the backpropagation computes the gradient of the loss function with respect to the weights of the network for a single input–output example,and updates weights to minimize loss.

## Data

### Zoo Dataset

This dataset consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals. The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate. The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables.

Attribute Information: (name of attribute and type of value domain)

1.	animal_name: Unique for each instance
2.	hair Boolean
3.	feathers Boolean
4.	eggs Boolean
5.	milk Boolean
6.	airborne Boolean
7.	aquatic Boolean
8.	predator Boolean
9.	toothed Boolean
10.	backbone Boolean
11.	breathes Boolean
12.	venomous Boolean
13.	fins Boolean
14.	legs Numeric (set of values: {0,2,4,5,6,8})
15.	tail Boolean
16.	domestic Boolean
17.	catsize Boolean
18.	class_type Numeric (integer values in range [1,7])

Download link: https://www.kaggle.com/uciml/zoo-animal-classification?select=zoo.csv

### The CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Download link: http://www.cs.toronto.edu/~kriz/cifar.html

## Algorithm

Generally, The network model consists of following conponents.
~~
 1. input (vector of features)
 
 2. target output
   For classification, output will be a vector of class probabilities, and target output is a specific class.
   
 3. loss function or "cost function
 
 4. the number of layers
 
 5. the weights between layer and the weight between the nodes in one layer.
 
 6. activation functions at layer
~~
### Animal classification
~~
1. Input Layers

  The width of the input layer was chosen to match the number of features for each dataset. The input layer was given a width of 16, since it has 16 features

2. Hidden Layers

  The hidden layer width was arbitrarily set to size 20.

3. Output Layers

  The width of the output layer was set to match the number of possible classes for a given observation. The output layer width was set to size 7, since its network is attempting to correctly classify an animal to its given class (mammal, bird, reptile, fish, amphibian, bug, invertebrate). 

4. Activation functions

  The rectified linear unit (ReLU) was chosen as the activation functions for the hidden layer. ReLUs perform fast calculations and are known to avoid the vanish gradient problem that occurs with logistic sigmoid activation functions.
  The softmax activation function was chosen for the output layer. This is because softmax functions output probablistic distributions for a given set of classes, which is useful for multiclass classification problems.

5. Training and Testing

  Roughly 80% of the data was used to train the model, while the other 20% was used for testing. Both models were then trained for a length of 10000 epochs.
~~
Discussion
The neural network model for the zoo dataset had an accuracy of 80%. This model finished training much faster than the model used for the mushroom dataset because the computations for the derivative of the ReLU activation function are much quicker than those for the derivative of the sigmoid activation function.


## Approach
