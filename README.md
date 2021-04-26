# Says One Neuron To Another

  EECS738 (Spring 2021) assignment 3

  Author: Ye Wang

## Purpose

The purpose of this project is to understand the underlying mathematics of neural networks, especially, to understand how the backpropagation computes the gradient of the loss function with respect to the weights of the network, and updates weights to minimize loss.

Animal_Classification is used to build a simple neural network to deal with simplified data in csv format. 

Image_Classification is used to build a more complex neural network to deal with images. 

## Data

### Zoo Dataset

This dataset consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals. The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate. The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables.

Part of attribute information:

1.	hair Boolean
2.	feathers Boolean
3.	eggs Boolean
4.	milk Boolean
5.	airborne Boolean
6.	aquatic Boolean

Download link: https://www.kaggle.com/uciml/zoo-animal-classification?select=zoo.csv

### The CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Download link: http://www.cs.toronto.edu/~kriz/cifar.html

## Algorithm


Generally, the network model consists of the following components.

 1. input (vector of features)
 
 2. target output
   For classification, the output will be a vector of class probabilities, and the target output is a specific class.
   
 3. loss function or "cost function
 
 4. the number of layers
 
 5. the weights between layers and the weight between the nodes in one layer.
 
 6. activation functions at each layer


### Animal classification


1. Input Layers

    The width of the input layer was chosen to match the number of features for each dataset. The input layer was given a width of 16 since it has 16 features.

2. Hidden Layers

    The hidden layer width was arbitrarily set to size 32.

3. Output Layers

    The width of the output layer was set to match the number of possible classes for a given observation. The output layer width was set to size 7 since its network is attempting to correctly classify an animal to its given class (mammal, bird, reptile, fish, amphibian, bug, invertebrate). 

4. Activation functions

    The rectified linear unit (ReLU) was chosen as the activation function for the hidden layer. ReLUs perform fast calculations and are known to avoid the vanish gradient problem that occurs with logistic sigmoid activation functions.
  
    The softmax activation function was chosen for the output layer. This is because softmax functions output probabilistic distributions for a given set of classes, which is useful for multiclass classification problems.

5. Training and Testing

    Deal with the raw data in _main.py_, and split the data into two parts, roughly 70% of the data was used to train the model, while the other 30% was used for testing. The ratio can be adjusted through argument in _split = round ()_ of the _main.py_.
  
    The feed forward function is achieved in the _NN.py_ by _mlp_ function, and the backpropagation function is achieved by _cross_entropy_back_prop_ function.
  
    The _model_accuracy_ function is used to calculate the accuracy in each iteration and guide the updating of the model parameters to meet the loss function.
  

### Image classification

The neural network framework is shown as below:

![Alt text](https://github.com/yeah61/Says-One-Neuron-to-Another/blob/main/Image_Classification/model%20framework.png)

## Approach

### Animal classification

#### prerequisite

   This project was written in Python 3.7.2 and relies on two modules, **numpy** and **pandas**.

#### train 

   Run the following command to train the NN model:

```
python3 main.py
```

   Modify the arguments in the _split = round()_ to change the ratio of training and testing data.

   Modify the arguments in the _model = mlp()_ to change the hidden layer width, learning rate and maximum epochs.

### Image classification

#### prerequisite

   This project was written in Python 3.7.2 and relies on one module **numpy** .

#### train 

   Run the following command to train the NN model:

```
python3 start.py
```
## Results

### Animal classification

![Alt text](https://github.com/yeah61/Says-One-Neuron-to-Another/blob/main/Animal_Classification/animal_classification_result.png)

The neural network model for the zoo dataset had an accuracy of 80%. This model finished training very fast because the computations for the derivative of the ReLU activation function are much quicker than those for the derivative of the sigmoid activation function.

Try training the model with different hidden layer widths, but the accuracy doesn't change.

### Image classification

![Alt text](https://github.com/yeah61/Says-One-Neuron-to-Another/blob/main/Image_Classification/Image_classification_result.png)

The accuracy is only 0.14, the main reason may be: 

1. the model is too simple with only one convolution, Feature extraction cannot reflect real features

2. overfitting

3. original train data is too complex and comprehensive

## Acknowledgement

Through this project, I get a basic understanding of the neural network. But there is still a long way to utilize the neural network model to do more complex tasks.
such as the image classification in this repository. I made a lot of references to this article: 
https://www.cnblogs.com/further-further-further/p/10430073.html. I am looking forward to further study and practice to get a better understanding of the neutral network.
