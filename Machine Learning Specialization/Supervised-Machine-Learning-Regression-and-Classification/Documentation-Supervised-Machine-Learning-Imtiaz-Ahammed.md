# Supervised Learning Part 1

It refers to such Algorithm that tries to learn and map the input(x) to output(y).

Example of different ML Application in Real Life:

| Input (x)        | Output (y)             | Application          |
| ---------------- | ---------------------- | -------------------- |
| Email            | Spam/Not-Spam(0/1)     | Email Classification |
| Audio            | Text Transcript        | Speech Recognition   |
| English          | Spanish                | Machine Translation  |
| Ad/User Info     | Click/No-Click (0/1)   | Online Advertising   |
| Image/Radar Info | Position of other cars | Self-Driving Car     |
| Images of Phone  | Defect/No-Defect(0/1)  | Visual Inspection    |


Regression is one kind of supervised learning algorithm that max x to y. More specifically that tries to find the linear trend in the dataset using the equation y=mx+c. This algorithm is mainly used to predict number for example predicting House Price from HousePrice dataset. This algorithm is mainly used 

# Supervised Learning Part 2

Classification is type of supervised learning algorithm that tries to classify different classes depending on the structure of the dataset. Type of Classification Problem:  
1. Binary Classification(0/1)
2. Multi-Class Classification(1/2/3...) 

![CLassification Visualization](![image](https://github.com/user-attachments/assets/6b8df260-275f-41b0-ac46-b10250970474)

This represents a classification task which mainly differentiate between two cancer classes

* Malignant
* Benign

The prediction of the classification or the output of the classification problem can be numeric or non-numeric. But the output can't be any float numbers or large like Regression Algorithm.


# Unsupervised Learning Part 1 and 2

It refers to algorithms that learns from the dataset which is not labeled. In other words, it tries to find the trend or pattern in a given dataset.

Clustering is one of most used unsupervised learning algorithm that tries to custer or group the dataset. In this algorithm, dataset only comes with input x, but not with output label. The algorithm tries to find the structure or trend between the dataset.

Example: 
1. Google news tries to gather similar types of news from all over the internet that has words or context in common. 

2. DNA Microarray uses clustering to find specific pattern in the DNA array and tries to group them into different classes.
   
3. Anomaly Detection: It tries to find the datapoint that are mainly unusual or doesn't follow the major trend of the dataset. 

4. Dimensionality Reduction: It compress the data using fewer numbers.s

# Standard Notation in Linear Regression

y = f(x) = wx + b

x = input/features variable
w = Bias/weights
b = bias
y = output/target variable

y_hat = Output/Prediction of model

![alt text](![image-1](https://github.com/user-attachments/assets/42bc1207-2f3c-417a-a3ea-706fbac9c2a9)

![Linear Regression with 1 variable](![image-2](https://github.com/user-attachments/assets/d4c162b9-4017-466e-b298-f52546f9a55b)

# Cost Function 
It is used to measure the performance or error of a model. It is also known as Squared Error Cost Function. Cost function is denoted as J(w,b) which is function of w, b. It tries to find difference between the actual values and the predicted values from mode.

![Cost Function J](![image-3](https://github.com/user-attachments/assets/d5ad22fd-0051-4d3d-8c13-43520f8dce66)
![alt text](![image-4](https://github.com/user-attachments/assets/8108759f-1e0e-4d60-ada6-5f6c68a9b3b7)

The cost function tries to minimize the error value as low as possible to find the best fit model.

![3D Plot of Cost Function](![image-5](https://github.com/user-attachments/assets/6f982bf5-6416-42a3-8f81-97df46ac5c96)

This plot shows the cost function with respect to w and b on the x and y axis respectively. On the z axis there is the output of the function J(w,b). While model training the goal of the model is to minimize the cost function.

While using squared error function as cost function with linear regression there will not be multiple local minima. It happens due to the convex property of the cost function squared error, which has and will be only 1 local minima as the function is bowl shaped function. 

# Gradient Descent

It is an algorithm that tries to minimize the value of the cost function by finding the global minima.

Basic Steps of Gradient Descent:

* Start with random w and b
* Calculate output of the cost function.
* Adjust the cost function to reduce the output of cost function until we find or settle at near minimum

![Parameters Update in Gradient Descent](![image-6](https://github.com/user-attachments/assets/0310525c-5859-4608-b917-362b36eb94ec)

![Finding Minima Using Gradient Descent](![image-7](https://github.com/user-attachments/assets/e4e44533-a290-4949-8350-b31ed73aaaba)

![Equation related to Linear Regression](![image-8](https://github.com/user-attachments/assets/355839f1-6a13-4cac-a5c3-ebd7a26b045d)


Type of Gradient Descent:
1. Batch Gradient Descent: Here, the model gets all the training data at once. In other words, the model process all the training example in each steps.
2. Stochastic Gradient Descent:
3. Mini-Batch Gradient Descent:

# Learning Rate(alpha)

The term alpha in gradient descent decide how much w and b will change in each step. In other words, it will try to minimize the parameters by taking steps. However, if learning rate is too low, it might take longer time to find the minima. On the other hand, if the learning rate is too high the model will jump abruptly and end up not being finding the local minima. So, we have to choose the value of alpha in between. 

# Multiple Linear Regression
In multiple linear regression we have multiple features like X1, X2, X3 and so on. So, the equation for multiple linear regression will be:

f(w,b) = W1X1 + W2X2 + W3X3 + ..+ WnXn.. + b

W = [W1 W2 W3 .... Wn]
X = [X1 X2 X3 .... Xn] 

Here, W is a vector that contains multiple weights as  vectors. Same goes for X as well.

![Function Equation with code](![image-9](https://github.com/user-attachments/assets/9694eb33-bab8-4be6-a967-22ec1aff66d3)


```python 
#without vectorization
f = 0
for j in range(0, n):
    f = f + w[j] * x[j]
f = f + b    
```

```python 
#with vectorization
f = np.dot(w*x)+b  
```
This will do the element wise dot product of w and x. It is more efficient compared to the previous one. 

![vector process](![image-10](https://github.com/user-attachments/assets/da82fec4-48c3-4bda-b6a0-c892aa508e8c)


# Vector Implementation of Gradient Descent

Array operations specially vector operations can run faster compared to normal operations. Furthermore, it uses gpu to run these computations in parallel.

```python
#without vectorization
for j in range (0, 15)
f=f+w[j]*x[j]
```
![alt text](![image-11](https://github.com/user-attachments/assets/11fb9190-cfbb-4cdc-9474-b6caf625ee33)

# Alternative to Gradient Descent
Normal equation is an alternative to gradient descent. 

Pros:
* It can only be used for linear regression
* Solve for w and b without iterations

Cons:
* Doesn't generalize to other learning algorithms
* It becomes slow when working with large dataset.


