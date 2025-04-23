# Supervised Learning Part-1

A type of machine learning that learns from given examples. The model takes the examples as input x, and their corresponding labels as y. The model uses a learning algorithm to learn from these x and y pairs so that, it can generate close to accurate y outputs for an input x that was never seen by the model before.

Regression and Classification are two major types of supervised learning algorithms.

# Supervised Learning Part-2

While in regression algorithms output numbers from numerical inputs, in classification the learning algorithm has to make a decision for the best suited category from the available categories.

# Unsupervised Learning


Unsupervised learning doesn't provide labels with each input data for the model to learn from. Instead the algorithm learns patterns and identifies different categories or groups present in the data and cluster them. So, next time when it encounters an unseen data, the algorithm compares it with the existing groups or clusters and recognize it as the cluster it is closed to in terms of patterns. 

Another type of unsupervised learning is anomaly detection, where unusal activities are detected.

# Linear Regression

In supervised learning the model is provided with a training set where it has features and corresponding target variables. The learning algorithm then produce a function. The purpose of this function is to take inputs and generate an output which is going to be a prediction.

![linear_regression](https://github.com/user-attachments/assets/f2895564-3f79-417e-a6b4-888a97d924ac)

# Cost Function

The cost function determines how well the model is doing.

![cost_function_1](https://github.com/user-attachments/assets/007fb2fd-66cc-40d7-952c-09590c762902)


# Squared Error Cost Function

The cost function takes the distance between the predicted output and actual output and tries to minimize it.

![cost_function_2](https://github.com/user-attachments/assets/b7563ad3-8e3e-4673-a15b-7a731254c06b)

To minimize the error, the distance is squared and then this squared error is computed across all of the training examples and summed up. To ensure that the cost function doesn't get bigger as the traininng set expands, it is a convention to divide the summation of the error with the number of training examples. Which makes the avereged squared error. For better computation the summation is divided by 2 times m, m is the number of training set examples.

The main goal of linear regression is to find the paramaters that will result in the smallest possible values for the cost function J. This way the 
cost function gets minimized. The image below shows the cost function while setting one parameter 0. 


![linear_regression](https://github.com/user-attachments/assets/f2895564-3f79-417e-a6b4-888a97d924ac)

