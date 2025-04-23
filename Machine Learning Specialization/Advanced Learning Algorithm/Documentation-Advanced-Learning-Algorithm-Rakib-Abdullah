# Week 1 (Neural Network) 

- As the amount of the data increased over time,classical models could not keep up and therefore,deep learning networks came

## Overview of neural network

- think of it as-each layer kind of works as a logistic regression but it can derive its new features on its own

- number of hidden layers and number of neurons on each layer depends on the architecture of the neural network.Performance also depends on them. 

- Multilayer perceptron

![image.png](attachment:image.png)

## An example : How nueral network recognizes images

![image-2.png](attachment:image-2.png)

## layer structure and function

![image-3.png](attachment:image-3.png)

![image-4.png](attachment:image-4.png)

- remember,correct input to a neuron is a vector

![image-5.png](attachment:image-5.png)

## Forward Propagation

![image-6.png](attachment:image-6.png)

## Inference in Code



![image-7.png](attachment:image-7.png)

## TensorFlow and inference in Code

Remember this code is not using tensorFlow pipelines

![image-8.png](attachment:image-8.png)

## Matrices in NumPy

![image-9.png](attachment:image-9.png)

## TensorFlow way of running an inference

![image-10.png](attachment:image-10.png)

## Forward prop in a single layer

![image-11.png](attachment:image-11.png)

- the code of forward prop in numpy from scratch: 

![image-12.png](attachment:image-12.png)

## Vectorized implementation of the same code above

- it uses matrix multiplication

![image-13.png](attachment:image-13.png)

# Week 2 (Neural Network Training)

- basic code for training 

![image.png](attachment:image.png)

## Model tranining steps

- the fit function does the forward and back propagation

![image-2.png](attachment:image-2.png)

## Commonly used Activation Function

![image-3.png](attachment:image-3.png)

## How to choose activation function for the output layer

## Output Layer

- Sigmoid -> used for binary classification problem
- Linear activation function -> Regression problem when value can be both positive and negative
- ReLu -> Regression but value can be only positive

## Hidden layer

- ReLU is the most common one

![image-4.png](attachment:image-4.png)

## Why we don't we use linear activation function everywhere in the nueral network

- if we do so,then the output will also become linear
- even if we use sigmoid function in the output layer it will act like a logistic regression model,nothing new or complex

![image-5.png](attachment:image-5.png)

## Multiclass classification problem with Softmax

- we use softmax regression for multiclass classification
- logistic regression vs softmax regression
- in this case we use multiple output unit(like 10 output unit for 10 classes),previously we used
  1 unit for binary classes in the output classes
- 1 key thing to note here is that,in logistic regression output ai is function of zi,but in
  case of softmax,ai is function of z1.......zn

![image-6.png](attachment:image-6.png)

- **Important** : How to calculate loss and cost </b>

- if we want to calculate for loss of y=2,we calculate only this -log(a2)

![image-7.png](attachment:image-7.png)

![image-8.png](attachment:image-8.png)

#3Code for this</h4>

![image-9.png](attachment:image-9.png)

## Multilabel classification Problem

- main idea : one input -> multilabel output(in multiclass output is one having multiple possibilities)

![image-10.png](attachment:image-10.png)

![image-11.png](attachment:image-11.png)


## Advanced Optimization of Gradient Descend in Neural Network

## Adam algorithm

- adapts the learning rate 

![image-12.png](attachment:image-12.png)

## In Code 

![image-13.png](attachment:image-13.png)

## Additional Layer(Convolutional Layer)

- each unit in the layer only look at a portion of the input

![image-14.png](attachment:image-14.png)

- in practice : 

![image-15.png](attachment:image-15.png)

<h1>Week 3</h1>
<h2>Machine learning Diagnostic</h2>

- Evaluating Machine Learning Models : split dataset into training and testing

![image.png](attachment:image.png)

![image-2.png](attachment:image-2.png)

- another way for classification problem

![image-3.png](attachment:image-3.png)

- Training error is not a good indicator


<h4>A better way to choose between models</h4>

- Training , validation and testing set
- choose model with lower cross validation error when choosing between multiple models
  so choose model based on how much good it performs on validation set

![image-4.png](attachment:image-4.png)

<h4>How to determine bias/variance</h4>

![image-5.png](attachment:image-5.png)

![image-6.png](attachment:image-6.png)

- the below example contains both high bias and high variance

![image-7.png](attachment:image-7.png)

- Choosing a good value of lemda during regularization for low bias and variance

![image-8.png](attachment:image-8.png)

<h4>How lamda affects loss</h4>

- higher lamda,lower w value,higher training loss,lower lamda higher w value,lower training loss


![image-9.png](attachment:image-9.png)

<h3>Establishing a baseline level of performance</h3>

![image-10.png](attachment:image-10.png)

![image-11.png](attachment:image-11.png)


<h3>Learning Curve</h3>

![image-12.png](attachment:image-12.png)

<h4>Larger training set does not help model to fix high bias problem,but it helps in case of high variance problem</h4>

![image-13.png](attachment:image-13.png)

![image-14.png](attachment:image-14.png)

<h3>Fixing High bias and variance problem summary</h3>

![image-15.png](attachment:image-15.png)

<h3>Bias Variance in neural network</h3>

![image-16.png](attachment:image-16.png)

- larger neural network does not mean that it will increase variance if regularization is ok,larger neural network is a low bias machine

![image-17.png](attachment:image-17.png)

<h2>Machine Learning Development Process</h2>

![image-18.png](attachment:image-18.png)

- Error Analysis (spam email classifier example) : this is also important beside
  bias and variance analysis

![image-19.png](attachment:image-19.png)

<h4>Adding more data</h4>

- focuses on the data that were focused by error analysis
- data augmentation works in case of images or for speech recognition
- but remember don't add noise to the data that are meaningless 

![image-20.png](attachment:image-20.png)

<h3>Data synthesis</h3>

- Using artificial data inputs to create a new training examples

![image-21.png](attachment:image-21.png)

<h3>Transfer Learning</h3>

![image-22.png](attachment:image-22.png)

![image-23.png](attachment:image-23.png)

<h2>Full Cycle of machine learning project</h2>

![image-24.png](attachment:image-24.png)

![image-25.png](attachment:image-25.png)

<h3>Skewed Datasets</h3>

- for skewed datasets,we use precision and recall when we have a rare class

![image-26.png](attachment:image-26.png)

<h4>Precision Recall tradeoff</h4>

![image-27.png](attachment:image-27.png)

- clear explaination of the above picture

![image-28.png](attachment:image-28.png)

<h3>Balancing precision and recall with f1 score</h3>


![image-29.png](attachment:image-29.png)





