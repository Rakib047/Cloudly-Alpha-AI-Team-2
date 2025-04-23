# Week 1 (Neural Network) 

- As the amount of the data increased over time,classical models could not keep up and therefore,deep learning networks came

## Overview of neural network

- think of it as-each layer kind of works as a logistic regression but it can derive its new features on its own

- number of hidden layers and number of neurons on each layer depends on the architecture of the neural network.Performance also depends on them. 

- Multilayer perceptron

<img width="656" alt="image" src="https://github.com/user-attachments/assets/b7f05a46-61c1-436c-8850-a6eb7a48857f" />


## An example : How nueral network recognizes images

<img width="658" alt="image" src="https://github.com/user-attachments/assets/f1e69046-ddd1-4ddf-95ce-9db292bb37cf" />


## layer structure and function

<img width="659" alt="image" src="https://github.com/user-attachments/assets/3b6ed172-7bef-49bc-80cf-a58e014ebc7f" />


<img width="656" alt="image" src="https://github.com/user-attachments/assets/79cf4938-fb0f-49ea-b8a4-36e1d67dc928" />


- remember,correct input to a neuron is a vector

<img width="657" alt="image" src="https://github.com/user-attachments/assets/83da7c5f-f0a7-4cad-955e-c58eb5f18f1e" />


## Forward Propagation

<img width="661" alt="image" src="https://github.com/user-attachments/assets/76cddb17-5f79-43fc-885a-df902bed0d53" />


## Inference in Code



<img width="657" alt="image" src="https://github.com/user-attachments/assets/3ccf7c2b-3724-40c2-8440-772e059ecf8f" />


## TensorFlow and inference in Code

Remember this code is not using tensorFlow pipelines

<img width="658" alt="image" src="https://github.com/user-attachments/assets/3d6bfe4c-c6fa-4481-beb2-d7db4fc65e90" />


## Matrices in NumPy

<img width="656" alt="image" src="https://github.com/user-attachments/assets/a90c4650-8ca4-4353-a571-3290aa3bf51e" />


## TensorFlow way of running an inference

<img width="657" alt="image" src="https://github.com/user-attachments/assets/be2e6eab-34b1-4191-ad4f-cac71b5b2aea" />


## Forward prop in a single layer

<img width="659" alt="image" src="https://github.com/user-attachments/assets/6b72ff79-ed40-4a82-a71d-f77e7449bdf6" />


- the code of forward prop in numpy from scratch: 

<img width="658" alt="image" src="https://github.com/user-attachments/assets/b5e690cc-b3c5-413d-b846-0d06f3d8c4b4" />


## Vectorized implementation of the same code above

- it uses matrix multiplication

<img width="657" alt="image" src="https://github.com/user-attachments/assets/25780148-ce27-4766-9e56-657119f0a9ad" />


# Week 2 (Neural Network Training)

- basic code for training 

<img width="657" alt="image" src="https://github.com/user-attachments/assets/f3a5fb8d-b24b-46e3-b7c1-e8c4ed992993" />


## Model tranining steps

- the fit function does the forward and back propagation

<img width="657" alt="image" src="https://github.com/user-attachments/assets/ac8c054c-6205-443a-b233-ad5ccaa264a5" />


## Commonly used Activation Function

<img width="657" alt="image" src="https://github.com/user-attachments/assets/86b545de-69bf-4483-8827-88eaf3f16258" />


## How to choose activation function for the output layer

## Output Layer

- Sigmoid -> used for binary classification problem
- Linear activation function -> Regression problem when value can be both positive and negative
- ReLu -> Regression but value can be only positive

## Hidden layer

- ReLU is the most common one

<img width="660" alt="image" src="https://github.com/user-attachments/assets/a7d8a277-377b-4ec5-91f8-350f875140f4" />


## Why we don't we use linear activation function everywhere in the nueral network

- if we do so,then the output will also become linear
- even if we use sigmoid function in the output layer it will act like a logistic regression model,nothing new or complex

<img width="659" alt="image" src="https://github.com/user-attachments/assets/d88340ec-6ef1-490e-b387-4d0410abbea7" />


## Multiclass classification problem with Softmax

- we use softmax regression for multiclass classification
- logistic regression vs softmax regression
- in this case we use multiple output unit(like 10 output unit for 10 classes),previously we used
  1 unit for binary classes in the output classes
- 1 key thing to note here is that,in logistic regression output ai is function of zi,but in
  case of softmax,ai is function of z1.......zn

<img width="657" alt="image" src="https://github.com/user-attachments/assets/647e2daf-054f-4f94-abec-19ba8bc43ad9" />


- **Important** : How to calculate loss and cost </b>

- if we want to calculate for loss of y=2,we calculate only this -log(a2)

<img width="657" alt="image" src="https://github.com/user-attachments/assets/2c13f7a5-1b3e-4a9e-be26-b21dbeeb8163" />


<img width="657" alt="image" src="https://github.com/user-attachments/assets/e810c772-c59f-4337-9f5d-34480c09b102" />


#3Code for this</h4>

<img width="658" alt="image" src="https://github.com/user-attachments/assets/1a28054a-4c5b-4043-b893-322eb5f677fa" />


## Multilabel classification Problem

- main idea : one input -> multilabel output(in multiclass output is one having multiple possibilities)

<img width="658" alt="image" src="https://github.com/user-attachments/assets/6a369bd0-af70-441f-b1a4-68aaea0cf9f7" />


<img width="660" alt="image" src="https://github.com/user-attachments/assets/3039c480-3b19-4348-bbe2-04bf268fcd77" />



## Advanced Optimization of Gradient Descend in Neural Network

## Adam algorithm

- adapts the learning rate 

<img width="659" alt="image" src="https://github.com/user-attachments/assets/6cdb6364-c7ac-4647-b600-a14212bf78ca" />


## In Code 

<img width="660" alt="image" src="https://github.com/user-attachments/assets/19dd9623-d57c-40cb-b121-2a8587cc79dc" />


## Additional Layer(Convolutional Layer)

- each unit in the layer only look at a portion of the input

<img width="660" alt="image" src="https://github.com/user-attachments/assets/ebc87b72-0821-48fe-a2d5-53e78ac9125e" />


- in practice : 

<img width="660" alt="image" src="https://github.com/user-attachments/assets/b5f8ffcb-4be4-472b-8fa7-f674316dece9" />


# Week 3
## Machine learning Diagnostic

- Evaluating Machine Learning Models : split dataset into training and testing

![image.png](attachment:image.png)

![image-2.png](attachment:image-2.png)

- another way for classification problem

![image-3.png](attachment:image-3.png)

- Training error is not a good indicator


### A better way to choose between models

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


## Learning Curve

![image-12.png](attachment:image-12.png)

### Larger training set does not help model to fix high bias problem,but it helps in case of high variance problem</h4>

![image-13.png](attachment:image-13.png)

![image-14.png](attachment:image-14.png)

## Fixing High bias and variance problem summary

![image-15.png](attachment:image-15.png)

## Bias Variance in neural network

![image-16.png](attachment:image-16.png)

- larger neural network does not mean that it will increase variance if regularization is ok,larger neural network is a low bias machine

![image-17.png](attachment:image-17.png)

## Machine Learning Development Process

![image-18.png](attachment:image-18.png)

- Error Analysis (spam email classifier example) : this is also important beside
  bias and variance analysis

![image-19.png](attachment:image-19.png)

## Adding more data

- focuses on the data that were focused by error analysis
- data augmentation works in case of images or for speech recognition
- but remember don't add noise to the data that are meaningless 

![image-20.png](attachment:image-20.png)

## Data synthesis

- Using artificial data inputs to create a new training examples

![image-21.png](attachment:image-21.png)

## Transfer Learning

![image-22.png](attachment:image-22.png)

![image-23.png](attachment:image-23.png)

## Full Cycle of machine learning project

![image-24.png](attachment:image-24.png)

![image-25.png](attachment:image-25.png)

## Skewed Datasets

- for skewed datasets,we use precision and recall when we have a rare class

![image-26.png](attachment:image-26.png)

## Precision Recall tradeoff

![image-27.png](attachment:image-27.png)

- clear explaination of the above picture

![image-28.png](attachment:image-28.png)

<h3>Balancing precision and recall with f1 score</h3>


![image-29.png](attachment:image-29.png)





