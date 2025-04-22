# Week 1
### What is machine learning

- Supervised Learning 
- Unsupervised Learning 
- Recommender Systems 
- Reinforcement Learning 

### Supervised Machine Learning

- input and output is given
- labeled data
- learning to map the input to output more accurately
- data has right answers
- regression model and classification model

### Regression

- for continous value mapping
- infinite range of possible outputs

![alt text](image-3.png)

### Unsupervised Learning

- No labeled data

### Types

- Clustering
- Anomaly Detection
- Dimentionality Reduction

### Clustering

- group similar data points together
- finding structure or pattern in the data
- only x given,no y

![alt text](image-4.png)

### Regression Model

- these models predict numbers

### Linear Regression Model

- training set : Features+target 

- Linear regression with one variable(univariate) : y = Wx + B

### Cost function 

- function of parameters

![alt text](image-5.png)


![alt text](image-6.png)

- cost function visualization

![alt text](image-7.png)

### Contour Plot

![alt text](image-8.png)

### Gradient Descent

- Main goal is to reduce the value of cost function
- update the weight values so that we can reduce the value of the cost function
- simultenous updates of parameters

### The algorhtm and some mistakes 

![alt text](image-27.png)

### The learning rate and derivative term intution

- alpha too small,gradient descent will be slow
- alpha too big,gradient descent will overshoot the minimum
- the derivative term decides in which direction and how much we should take the next step

### Problem of local minima


- Near a local minimum,the derivative gets smaller and smaller as well as closer to zero,therefore,the updates becomes smaller and smaller again

![alt text](image-28.png)

### Gradient descent algorithm for linear regression

![alt text](image-29.png)

- Batch Gradient Descent : taking all examples for at each step of GD

# Week 2
### Multiple Linear Regression

- f(x)= w . x + b
      = (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4)..... + b

- vectorization in case of the  makes the computation much faster.For example, 
  f= np.dot(w,x) + b

### Gradient Descent for multiple Linear Regression 

![alt text](image-30.png)

**Final Form** : 

![alt text](image-31.png)

### Gradient Descend in Practice 

#### Reasons behind Scaling features within 0 and 1

- fast and smooth gradient descent
- 1st contuour->not good updates,but 2nd good

![alt text](image-32.png)

### Feature Scaling

- by dividing max value

![alt text](image-33.png)

- mean normalization

![alt text](image-34.png)

- Z-score normalization

![alt text](image-35.png)

### Rule of Thumb 

- Feature scaling actually helps fast gradient descent

![alt text](image-36.png)

### Making sure whether gradient descent is working or not

- we need to check whether loss function is decreasing or increasing or reached convergence

![alt text](image-37.png)

### How to choose learning rate

- don't choose a big learning rate
- choose small one,but this can slow down convergence
- increase learning rate step by step

![alt text](image-38.png)

### Feature Engineering 

![alt text](image-39.png)

### Polynomial regression 

![alt text](image-40.png)


# Week 3

### Why linear regression doesn't work for classification

- notice the green line(green line came due to the outlier),decision boundary shifted(which should not be)

![alt text](image-26.png)


### Logisitic regression

- pass the linear regression to the sigmoid function :

   z = w.x + b
   g(z) = 1/(1+e^-z)

- a probability value is the output of logistic regression,we use a threshold in this 
  case,for example above 50% is considered as 1

- sigmoid function is used here

![alt text](image-24.png)

![alt text](image-25.png)

### Decision Boundary for logistic regression</h3>

- when z=0,it is the decision boundary

![alt text](image-23.png)

more example:

![alt text](image-22.png)

### Cost function for logistic regression

- squared error cost is not ideal,because it generates non convex cost function which does not work well in gradient descent due to getting stuck in local optima

![alt text](image-21.png)

### Visualizing logistic loss function

- the choice of this loss function makes the overall loss function convex
- when actual y=1

![alt text](image-20.png)

- when actual y=0

![alt text](image-19.png)

### Simplified Loss function

![alt text](image-18.png)

### Gradient Descent for logistic Regression

![alt text](image-17.png)

### The problem of overfitting

- Underfitting -> High bias
- overfitting -> high variance
- generalization on unseen data
- for linear regression(fitting a curve)/classification(finding a decision boundary)

### Solutions:

- get more training data (in case of overfitting)
- feature selection(using subset of features)
- regularization (for example reduce the size of parameters)
- modified cost function with regularization

![alt text](image-16.png)

### The intution behind modified cost function with regularization

- reducing the value of w3 and w4 weights

![alt text](image-13.png)

- at first we don't know which features are important,so at the beginning we penalize all the 
  weights a little

- lamda balances the goal of fitting data and keeping wj small,setting a proper lamba 
  value is important for addressing the issue of overfitting and underfitting

- increasing lamda will decreases the value of weights

![alt text](image-14.png)

![alt text](image-15.png)


### Regularized Linear Regression

![alt text](image-12.png)

### Regularized Logistic Regression

![alt text](image-11.png)

![alt text](image-10.png)

### Why to use DT or not

![alt text](image-9.png)





