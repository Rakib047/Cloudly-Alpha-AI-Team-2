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

<img width="713" alt="image" src="https://github.com/user-attachments/assets/2b9ae9d5-8a8d-4717-9132-fc8b9d962f28" />


<img width="716" alt="image" src="https://github.com/user-attachments/assets/739b47ea-3118-421d-9e39-f5f1b47d27d3" />


- another way for classification problem

<img width="714" alt="image" src="https://github.com/user-attachments/assets/acdf2432-e7ba-46df-9ff4-dc13057248fe" />


- Training error is not a good indicator


### A better way to choose between models

- Training , validation and testing set
- choose model with lower cross validation error when choosing between multiple models
  so choose model based on how much good it performs on validation set

<img width="712" alt="image" src="https://github.com/user-attachments/assets/c9eee478-9dc6-4f27-b678-012450254b8a" />


<h4>How to determine bias/variance</h4>

<img width="713" alt="image" src="https://github.com/user-attachments/assets/18b045a2-bd68-459e-a1eb-90e942cec577" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/e78af470-73d8-44fe-b82d-e1a44b3020a0" />


- the below example contains both high bias and high variance

<img width="474" alt="image" src="https://github.com/user-attachments/assets/f44675ed-a0a5-4609-8f60-7d89bae44adc" />


- Choosing a good value of lemda during regularization for low bias and variance

<img width="711" alt="image" src="https://github.com/user-attachments/assets/144fc23e-e2a3-4864-bb78-3c2855ad1f1d" />


## How lamda affects loss

- higher lamda,lower w value,higher training loss,lower lamda higher w value,lower training loss


<img width="713" alt="image" src="https://github.com/user-attachments/assets/b5eae083-be6d-4cdb-b400-ef672331834b" />


<h3>Establishing a baseline level of performance</h3>

<img width="713" alt="image" src="https://github.com/user-attachments/assets/0ffcb46b-718a-4789-bffa-8c2a238d1efc" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/c3152043-00d6-4197-a605-8a8dbdbbe3c2" />



## Learning Curve

<img width="713" alt="image" src="https://github.com/user-attachments/assets/4ebfd8f9-4e04-4dca-894a-1c865f89c21d" />


### Larger training set does not help model to fix high bias problem,but it helps in case of high variance problem</h4>

<img width="713" alt="image" src="https://github.com/user-attachments/assets/9585b9fb-b114-4db8-8559-ee8f1e6684a1" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/9f92f6f0-a202-45a9-b842-31f5f10dc2f9" />


## Fixing High bias and variance problem summary

<img width="713" alt="image" src="https://github.com/user-attachments/assets/6490f423-5bd7-41f1-8b4a-f22a34854c09" />


## Bias Variance in neural network

<img width="713" alt="image" src="https://github.com/user-attachments/assets/1429a75c-9d75-4117-93fa-c17e5b4c4396" />


- larger neural network does not mean that it will increase variance if regularization is ok,larger neural network is a low bias machine

<img width="713" alt="image" src="https://github.com/user-attachments/assets/808d1823-6c73-49e3-a61a-0ff294b5ebc0" />


## Machine Learning Development Process

<img width="713" alt="image" src="https://github.com/user-attachments/assets/0390bf14-3621-421a-8bcf-681e1f42fc8c" />


- Error Analysis (spam email classifier example) : this is also important beside
  bias and variance analysis

<img width="713" alt="image" src="https://github.com/user-attachments/assets/10fb5b82-dfc0-4ac9-9d29-196507ff6ff0" />


## Adding more data

- focuses on the data that were focused by error analysis
- data augmentation works in case of images or for speech recognition
- but remember don't add noise to the data that are meaningless 

<img width="710" alt="image" src="https://github.com/user-attachments/assets/39d22061-56a1-4a6f-9630-142b95f331b9" />


## Data synthesis

- Using artificial data inputs to create a new training examples

<img width="713" alt="image" src="https://github.com/user-attachments/assets/734bfd99-a8ed-42f1-9f74-e16acb50a725" />


## Transfer Learning

<img width="713" alt="image" src="https://github.com/user-attachments/assets/647d1be6-2d1b-45cc-a0a9-89d14998fd6b" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/20202297-e5dc-4b45-981b-dc995cb1dbc5" />


## Full Cycle of machine learning project

<img width="713" alt="image" src="https://github.com/user-attachments/assets/118e5898-fff0-4a6e-b293-7450b4bf3b7e" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/197e9c80-1670-4b98-a00e-18b06390fd17" />


## Skewed Datasets

- for skewed datasets,we use precision and recall when we have a rare class

<img width="713" alt="image" src="https://github.com/user-attachments/assets/19e5ef04-66ec-4fa8-8e16-63309428b7a7" />


## Precision Recall tradeoff

<img width="713" alt="image" src="https://github.com/user-attachments/assets/181c712c-2095-45cd-9b31-d82d467a4a27" />


- clear explaination of the above picture

<img width="712" alt="image" src="https://github.com/user-attachments/assets/7ec769e1-3c18-4dbc-aeb9-fd4eb5d932b2" />


<h3>Balancing precision and recall with f1 score</h3>


<img width="713" alt="image" src="https://github.com/user-attachments/assets/c461b8a3-fb8e-4847-b841-3868d406edff" />

# Week 4
## Decision Tree Model

<img width="713" alt="image" src="https://github.com/user-attachments/assets/7f800d94-8c5c-40a1-b4ff-a9d86d6d0e44" />


- This is an example of Decision tree,there can be many other combinations,our goal is to pick a decision tree

## How a decision Tree learns

<img width="713" alt="image" src="https://github.com/user-attachments/assets/b9a11095-e9c0-412f-af09-133fe3c820be" />


<img width="713" alt="image" src="https://github.com/user-attachments/assets/07cc5c1e-4d35-45f9-a2ef-05829fa5320f" />


## Entropy as a measure of impurity</h3>

- the higher the value of entropy,the higher the impurity

<img width="713" alt="image" src="https://github.com/user-attachments/assets/6aedf2e3-0de1-4abe-869b-db66e26ffe29" />


## The formula

<img width="715" alt="image" src="https://github.com/user-attachments/assets/fd75ebc0-23f4-4bd4-b520-d7c723cf980c" />


## Choosing what feature to choose during splitting,we use information gain</h4>

- ig indicates the reduction of entropy

<img width="713" alt="image" src="https://github.com/user-attachments/assets/644f6d09-ae58-436f-9934-a52307be38b9" />


## Decision Tree Learning Summary

<img width="713" alt="image" src="https://github.com/user-attachments/assets/4a77743f-9ca8-4432-b920-894e97e4afda" />


## One hot encoding</h3>

<img width="713" alt="image" src="https://github.com/user-attachments/assets/27c6cbe8-cbd8-4b80-9c72-152a8881de84" />


## Splitting for continuous value

- try out different values,check which one gives the highest information gain

<img width="713" alt="image" src="https://github.com/user-attachments/assets/3e48de77-2db0-419a-9f8b-13bf683e7b95" />


## Regression using DT

- we will choose that node that reduces variance the most

<img width="713" alt="image" src="https://github.com/user-attachments/assets/f40fea04-b28e-40f2-84c9-e9485dfb8fd5" />



## Tree Ensemble

- DT are highly sensitive to small changes of data

- here in the below picture we use 3 tree combined and used majority voting

<img width="713" alt="image" src="https://github.com/user-attachments/assets/857475c8-9319-45bf-a660-10c5701981a9" />


## Sampling with replacement

- some examples may repeat multiple times
- some examples may not even come for a single time in the training set
- for example given a dataset with 10 examples,place it in a virtual bag,create another training set of 10 examples by sampling with replacement
 
<img width="713" alt="image" src="https://github.com/user-attachments/assets/7838398f-ecad-41a5-a9f6-390152f69f48" />


## Random Forest

- build a training set with sampling with replacement..then train a DT on that set,do this process for example 100 times....this is the main thing

- for further randomization,instead of selecting from all features for node,we select one from random k features as our feature of choice

this is the core concept of random forest algorithm

## Boosted Decision Tree(XGBoost->Extreme gradient boosting)</h3>

- focus on more on misclassified examples

<img width="713" alt="image" src="https://github.com/user-attachments/assets/ead1bce2-5947-4138-baef-3ffdecb6aaab" />


### Why XGBoost

- fast,open sourced,good choice of default splitting criteria and criteria for when to stop
- built in regularization to prevent overfitting

<img width="709" alt="image" src="https://github.com/user-attachments/assets/579cdbfa-a845-4731-87df-75d6e5f81e7d" />







