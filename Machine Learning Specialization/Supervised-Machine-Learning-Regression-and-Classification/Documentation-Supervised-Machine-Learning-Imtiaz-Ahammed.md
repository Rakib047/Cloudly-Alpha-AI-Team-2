# Supervised Machine Learning: Regression and Classification 

## Topics Covered in Week 1

## Supervised Learning Part 1

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

## Supervised Learning Part 2

Classification is type of supervised learning algorithm that tries to classify different classes depending on the structure of the dataset. Type of Classification Problem:  
1. Binary Classification(0/1)
2. Multi-Class Classification(1/2/3...) 

![alt text](image.png)

This represents a classification task which mainly differentiate between two cancer classes

* Malignant
* Benign

The prediction of the classification or the output of the classification problem can be numeric or non-numeric. But the output can't be any float numbers or large like Regression Algorithm.

## Unsupervised Learning Part 1 and 2

It refers to algorithms that learns from the dataset which is not labeled. In other words, it tries to find the trend or pattern in a given dataset.

Clustering is one of most used unsupervised learning algorithm that tries to custer or group the dataset. In this algorithm, dataset only comes with input x, but not with output label. The algorithm tries to find the structure or trend between the dataset.

Example: 
1. Google news tries to gather similar types of news from all over the internet that has words or context in common. 

2. DNA Microarray uses clustering to find specific pattern in the DNA array and tries to group them into different classes.
   
3. Anomaly Detection: It tries to find the datapoint that are mainly unusual or doesn't follow the major trend of the dataset. 

4. Dimensionality Reduction: It compress the data using fewer numbers.s

## Standard Notation in Linear Regression

*y = f(x) = wx + b*

x = input/features variable
w = Bias/weights
b = bias
y = output/target variable

y_hat = Output/Prediction of model

![alt text](image-1.png)

![Linear Regression with 1 variable](image-2.png)

## Cost Function 
It is used to measure the performance or error of a model. It is also known as Squared Error Cost Function. Cost function is denoted as *J(w,b)* which is function of w, b. It tries to find difference between the actual values and the predicted values from mode.

![Cost Function J](image-3.png)
![alt text](image-4.png)

The cost function tries to minimize the error value as low as possible to find the best fit model.

![3D Plot of Cost Function](image-5.png)

This plot shows the cost function with respect to w and b on the x and y axis respectively. On the z axis there is the output of the function J(w,b). While model training the goal of the model is to minimize the cost function.

While using squared error function as cost function with linear regression there will not be multiple local minima. It happens due to the convex property of the cost function squared error, which has and will be only 1 local minima as the function is bowl shaped function. 

## Gradient Descent

It is an algorithm that tries to minimize the value of the cost function by finding the global minima.

Basic Steps of Gradient Descent:

* Start with random w and b
* Calculate output of the cost function.
* Adjust the cost function to reduce the output of cost function until we find or settle at near minimum

![Parameters Update in Gradient Descent](image-6.png)

![Finding Minima Using Gradient Descent](image-7.png)

![Equation related to Linear Regression](image-8.png)


Type of Gradient Descent:
1. Batch Gradient Descent: Here, the model gets all the training data at once. In other words, the model process all the training example in each steps.
2. Stochastic Gradient Descent:
3. Mini-Batch Gradient Descent:

## Learning Rate(alpha)

The term alpha in gradient descent decide how much w and b will change in each step. In other words, it will try to minimize the parameters by taking steps. However, if learning rate is too low, it might take longer time to find the minima. On the other hand, if the learning rate is too high the model will jump abruptly and end up not being finding the local minima. So, we have to choose the value of alpha in between. For learning rate alpha we can start from 0.001, then take the 3 times of the current value like 0.003.Thus, we will increase the or decrease value based on the impact of cost function to find a balanced number that is not is big and nor too small.

# Topics Covered in Week 2

## Multiple Linear Regression
In multiple linear regression we have multiple features like *X1, X2, X3* and so on. So, the equation for multiple linear regression will be:

*f(w,b) = W1X1 + W2X2 + W3X3 + ..+ WnXn.. + b*

*W = [W1 W2 W3 .... Wn]*
*X = [X1 X2 X3 .... Xn]* 

Here, W is a vector that contains multiple weights as  vectors. Same goes for X as well.

![Function Equation with code](image-9.png)

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
This will do the element wise dot product of **w* and *x*. It is more efficient compared to the previous one. 

![vector process](image-10.png)

## Vector Implementation of Gradient Descent

Array operations specially vector operations can run faster compared to normal operations. Furthermore, it uses gpu to run these computations in parallel.

```python
#without vectorization
for j in range (0, 15)
f=f+w[j]*x[j]
```
![alt text](image-11.png)

## Alternative to Gradient Descent
Normal equation is an alternative to gradient descent. 

Pros:
* It can only be used for linear regression
* Solve for w and b without iterations

Cons:
* Doesn't generalize to other learning algorithms
* It becomes slow when working with large dataset.

## Feature Scaling
Feature scaling is very much important steps in machine learning. Specially for the algorithm that uses distance. In dataset, multiple feature has different range. Some ranges from 0 to 100, whereas some ranges from 10 to 10,000. Impact of imbalanced feature range:

* **Slow Convergence:** When features have different scales, the cost function becomes skewed, causing gradient descent to take inefficient, zig-zag paths toward the minimum. This results in slower convergence.

* **Dominant Features**: If one or more features have significantly larger values than others, they can dominate the gradient updates, making it difficult for the algorithm to properly minimize the cost function. This can cause the model to fail to converge or oscillate without reaching the minimum.  

Feature Scaling Techniques:

1. Mean Normalization: It is also known as min-max scaling. It transform the features from range -1 to 1.

![Formula for Standardization=](<equation (1).png>)

2. Z Score Normalization: It transforms the feature such that they have mean = 0 and standard deviation of 1. 

![Formula for Z Score](equation.png)

## Learning Curve
It helps to understand how well our gradient descent algorithm is performing. To plot this curve we need to calculate the value of cost function after each iteration. If the Gradient Descent algorithm converges that means the graph will converge near to x axis with increase in iteration.

![alt text](image-12.png)

## Automatic Convergence Test
Let's take a number 0.001. If the cost function J(w,b) decreases by less than or equal 0.001 in one iteration, then we can say the model is converging. That means the cost function has found parameters to get close to global minima.

## Feature Engineering
It is most important task in machine learning while preparing the dataset for model. It refers to selecting important feature or creating new features from combining existing features that might help model in predicting more accurately.

## Polynomial Regression
It is a advance version of linear regression that uses higher order variable for fitting curved instead of straight line. The equation for polynomial regression can be described as following:

![alt text](image-13.png)

The term n decides the number of features to be fitted. It can vary based on dataset and requirements.Compared to only x it converts the feature power raise to n.

# Topics Covered in Week 3

## Classification using Linear Regression 
As, the linear regression model outputs number from negative infinity to positive infinity and continuous values like 1.5 -0.2, 2.5 and so on. So, it becomes challenging to use it for classification as it's output is not bounded in between 0 to 1. Also, regression model is sensitive to outliers.So, for classification task it will perform poorly. 
Even if we use threshold for example:
1. If *f(x)<0.5*, then prediction of the model will be  negative class 0.
2. If *f(x)>=0.5*, then prediction of the model will be positive class 1.

It will work well on dataset that can be separable linearly means the 2 classes can be separated using a straight line. But, as we know the output of the regression can be anything so threshold method will to classify the 2 classes.
   
![alt text](image-14.png)

## Logistic Regression
As we have seen the Linear Regression or any other regression model output is a discrete number. So it become difficult to classify in between 2 class.Even the structure or the distribution of the dataset might force the model to perform bad while classifying the classes. 

So, we need something which output will be always in between 0/1 no matter. To do this we require a new type of function called **Sigmoid/Logistic Function**.

## Sigmoid Function

![Sigmoid Function](image-15.png)

This is an exponential function which limits the output in between 0 to 1.
* When z is large positive number the output of g(z) is very close to 1.
* When z is large negative number the output of g(z) is very close to 0.
* When z=0, the output of g(z) is 0.5

![Logistic Regression Algorithm](image-16.png)

The above equation depicts how the linear regression model can be utilized for Binary Classification Task using the sigmoid function. The output of g(z) will always e in between 0 to 1.

The output of the sigmoid function along with linear regression can be interpreted as probability. For example:
* If the output of f(x)=0.7, then it means the probability for the given input to be positive class 1 is 70%.
* On the other hand, it also means that the probability for the given input to be negative class 1 is 30.

The probability of class 1/2 can be expressed as following:
![Log Reg Probability Eqn](image-17.png)
The equation means that probability of y to be class 1, given input x, parameters w and b.

## Decision Boundary for Classification
As we have seen, that the output of the sigmoid function along with linear regression can be any value between 0 to 1. So, we need to set a decision boundary for the model such that if the output falls below the given threshold it will be class 1 or vice versa. For example let's take a dataset with 2 features and 2 class to see how decision boundary work:

For 2 features along with 2 class the equation will be as following

![2 feature with 2 eqn](image-18.png)
* At point *z = 0* the model will be neutral.
* If *w1 = w2 = 1* and *b=-3*, then the equation will be as following:
  
![Decision Boundary](image-19.png)

If we plot the decision boundary to visualize how it works we can see that 1 linear line is separating both classes.

## Non-Linear Decision Boundary
As we have seen in the previous example that decision boundary was a linear line that separates 2 classes. But the decision boundary can also be non-linear. In that case, we need polynomial regression that has higher order terms to create non-linear decision boundary. Some example of non-linear decision boundary:

![Circle](image-20.png)

The decision boundary is circle as the equation is similar to a circle. So, the circle works as a boundary line in between 2 classes similar to the previous one.

![Complex](image-21.png)

The decision boundary becomes more complex with higher order polynomial terms in the equation.

## Cost Function for Logistic Regression

![Cost fn of LR](image-22.png)

From the cost function above we can see that the function is non-convex function. While dealing with an optimization problem with Non-convex graph we face the problem of getting stuck at the local minima instead of the global minima. The presence of multiple local minima can make it challenging to find the optimal solution for a machine learning model. If the model gets trapped in a local minimum, it will not achieve the best possible performance.   

So, we need a new cost function such that the it has the property of convex function and the gradient descent algorithm can optimize it during the training process.

![Logistic Loss Function](image-23.png)

1. *Positive Class:* When the model predicts 1 and the true label is also 1, then the loss is very minimum or close to 0(left most point of x-axis). On the contrary, if model predicts 0.1 but the actual class is 1. Then the loss is very much high(very upper point of the y-axis)
   
   * If the prediction is close to actual label the loss is close to 0
   * If the prediction is far from the actual label the loss is very high.

![log_reg=1](image-24.png)

1. *Negative Class:* When the model predicts 0 and the true label is also 0, then the loss is very minimum or close to 0(left most point of x-axis). On the contrary, if model predicts 0.7 or anything close to 1 but the actual class is 0, then the loss is very much high approaching to infinity(right most point of x axis)

* If the model approaches to 1 which is wrong prediction then the loss function will penalize the model with higher loss value.

![log_reg=0](image-25.png)

Thus, the different portion of the loss function mainly for class 1 and class 0 acts as convex function which is a must requirement for gradient descent to optimize model parameters minimizing the loss.

![2 plot of cost fn](image-26.png)

## Simplified Version of Loss & Cost Function

![simplified loss function](image-27.png)

From the above image, we can see that the piecewise function we have seen earlier can be written as a single function which actually works as before depending on the class selection(y=0/1).

![simplified cost function](image-28.png)

From the above image, we can see the cost function is nothing but the average of all loss function. However, this function was derived from a statistical method called Maximum Likelihood. It's a method tries that to find the best parameters for the model to make the model more probable in predicting the given dataset.

## Gradient Descent Equation for Logistic Regression

![LR_GD](image-29.png)

Here, the equation may look like same as we have seen in the linear regression. But, the definition of *f(x)* is not similar to linear regression as *f(x)* is passed to sigmoid function. The above 2 equations are used to update model parameters simultaneously while training the model. The steps we required for optimizing the gradient descent will be similar we have learnt in the linear regression part.

## Problem of Overfitting & Underfitting

It is very usual that sometimes machine learning model falls into problem like overfitting and underfitting.

![overfit-underfit problem](image-30.png)

1. *Underfit:* When model has high bias and unable to fit the training data well that is known as underfitting.
2. *Overfit:* When model has high variance and fits the training data very well that is known as overfitting.
3. *Generalization:* When model fits the training data almost close to actual or pretty well then that is known as generalization.

## Solution to Overfitting
1. Collect more data for training the model
2. Selecting relevant features 
3. Using regularization can shrink the large value of weights close to 0.

## Cost Function with Regularization 
As we need to apply regularization to prevent overfitting. We need to add a regularization term with the cost function.

![Cost Function with Regularization](image-31.png)

Choice of the lambda term select the impact of the regularization parameter.

1. If lambda is too high, the model tends to underfitting cause all the weight parameter value shrinks to 0.
   
2. If lambda is too low, like 0, the model tends to overfitting cause the weight value will be very high as there will be no impact on the weights.

## Regularization in Linear Regression

![Cost Fn with Regularization](image-32.png)


## Regularization with Logistic Regression

Equation for logistic regression with regularization including the cost function and parameter updating function as following

![LR_Reg](image-33.png)