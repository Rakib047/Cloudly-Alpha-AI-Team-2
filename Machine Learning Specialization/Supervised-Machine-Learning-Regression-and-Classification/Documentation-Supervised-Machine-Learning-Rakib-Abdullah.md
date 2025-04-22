<img width="806" alt="image" src="https://github.com/user-attachments/assets/9bfcf46e-70a1-417c-8e0b-9654019d6aaf" /># Week 1
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

<img width="806" alt="image" src="https://github.com/user-attachments/assets/7990c0a8-8aa2-46aa-9a42-ee33c8403225" />


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

<img width="789" alt="image" src="https://github.com/user-attachments/assets/e48ac5ec-402a-4547-9448-4ccc8e2496ed" />


### Regression Model

- these models predict numbers

### Linear Regression Model

- training set : Features+target 

- Linear regression with one variable(univariate) : y = Wx + B

### Cost function 

- function of parameters

<img width="794" alt="image" src="https://github.com/user-attachments/assets/00ec62c1-a1a6-4b9e-9cb1-cc2ca647b7a2" />



![alt text](image-6.png)

- cost function visualization

<img width="804" alt="image" src="https://github.com/user-attachments/assets/64330c49-4e63-4e30-8c38-f0a4c7a2b2d3" />
<img width="804" alt="image" src="https://github.com/user-attachments/assets/f9aa6452-283c-4654-ba96-ae051f51ebcf" />

<img width="804" alt="image" src="https://github.com/user-attachments/assets/4444ef01-0b82-4633-87c2-bcab611ec8da" />



### Contour Plot

<img width="795" alt="image" src="https://github.com/user-attachments/assets/c3138e2b-61b2-4335-bc29-d0c8afd82913" />


### Gradient Descent

- Main goal is to reduce the value of cost function
- update the weight values so that we can reduce the value of the cost function
- simultenous updates of parameters

### The algorhtm and some mistakes 

<img width="811" alt="image" src="https://github.com/user-attachments/assets/2716374d-72e9-46e2-99ae-fd929891a93b" />


### The learning rate and derivative term intution

- alpha too small,gradient descent will be slow
- alpha too big,gradient descent will overshoot the minimum
- the derivative term decides in which direction and how much we should take the next step

### Problem of local minima


- Near a local minimum,the derivative gets smaller and smaller as well as closer to zero,therefore,the updates becomes smaller and smaller again

<img width="806" alt="image" src="https://github.com/user-attachments/assets/2effa1db-fbc7-46cc-a47c-f6a67a4b6e3d" />


### Gradient descent algorithm for linear regression

<img width="806" alt="image" src="https://github.com/user-attachments/assets/506ca540-4f38-4408-92ec-3bb338609941" />


- Batch Gradient Descent : taking all examples for at each step of GD

# Week 2
### Multiple Linear Regression

- f(x)= w . x + b
      = (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4)..... + b

- vectorization in case of the  makes the computation much faster.For example, 
  f= np.dot(w,x) + b

### Gradient Descent for multiple Linear Regression 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/9dbf0c0c-e7fe-44d7-87f9-d9d92c5b85af" />


**Final Form** : 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/3e11a2d6-a1e0-4dc1-9eca-206ea62094f5" />


### Gradient Descend in Practice 

#### Reasons behind Scaling features within 0 and 1

- fast and smooth gradient descent
- 1st contuour->not good updates,but 2nd good

<img width="806" alt="image" src="https://github.com/user-attachments/assets/29b3ab89-53ba-4d20-9133-ddf169f6a6df" />


### Feature Scaling

- by dividing max value

<img width="806" alt="image" src="https://github.com/user-attachments/assets/80cfefdc-c74a-415a-abb9-85a48e08aff9" />


- mean normalization

<img width="806" alt="image" src="https://github.com/user-attachments/assets/096e898b-64ce-4b67-b619-81d95b88400a" />


- Z-score normalization

<img width="806" alt="image" src="https://github.com/user-attachments/assets/56f53c5b-7723-4300-a316-3b4e8dc8354c" />

### Rule of Thumb 

- Feature scaling actually helps fast gradient descent

<img width="797" alt="image" src="https://github.com/user-attachments/assets/a2c68716-3c03-4282-a0bb-0d9e69dca192" />


### Making sure whether gradient descent is working or not

- we need to check whether loss function is decreasing or increasing or reached convergence

<img width="809" alt="image" src="https://github.com/user-attachments/assets/8c2078ac-0236-441f-ab4d-b78be2beb185" />


### How to choose learning rate

- don't choose a big learning rate
- choose small one,but this can slow down convergence
- increase learning rate step by step

<img width="806" alt="image" src="https://github.com/user-attachments/assets/54d45c3c-8577-4e59-a7ef-cd00ad1c5779" />


### Feature Engineering 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/f8c2e031-80c2-44ed-af43-e6e983e761d5" />


### Polynomial regression 

<img width="818" alt="image" src="https://github.com/user-attachments/assets/2bcb962c-560b-4f60-ab83-50e8bfc1dd6c" />



# Week 3

### Why linear regression doesn't work for classification

- notice the green line(green line came due to the outlier),decision boundary shifted(which should not be)

<img width="806" alt="image" src="https://github.com/user-attachments/assets/7c124050-602b-408a-9cee-0732e44a6888" />



### Logisitic regression

- pass the linear regression to the sigmoid function :

   z = w.x + b
   g(z) = 1/(1+e^-z)

- a probability value is the output of logistic regression,we use a threshold in this 
  case,for example above 50% is considered as 1

- sigmoid function is used here

<img width="806" alt="image" src="https://github.com/user-attachments/assets/d9d03740-5470-475f-9cbc-e3c441a33de4" />


<img width="806" alt="image" src="https://github.com/user-attachments/assets/709f397a-429b-4d02-ad69-60f88f812d9d" />


### Decision Boundary for logistic regression</h3>

- when z=0,it is the decision boundary

<img width="806" alt="image" src="https://github.com/user-attachments/assets/6fba9e78-63ab-4ca1-9d0e-68da864c14f3" />


more example:

<img width="806" alt="image" src="https://github.com/user-attachments/assets/7afb525a-ab8e-4593-8279-ee6d1ad4f59d" />


### Cost function for logistic regression

- squared error cost is not ideal,because it generates non convex cost function which does not work well in gradient descent due to getting stuck in local optima

<img width="803" alt="image" src="https://github.com/user-attachments/assets/383a927d-a32a-4f1c-bbe2-dd6b4e5615dd" />


### Visualizing logistic loss function

- the choice of this loss function makes the overall loss function convex
- when actual y=1

<img width="806" alt="image" src="https://github.com/user-attachments/assets/85fe96db-3b3e-4d3f-9108-4b9581c0fbdf" />


- when actual y=0

<img width="806" alt="image" src="https://github.com/user-attachments/assets/78f22240-3333-4586-b275-b1ca3a94dcd9" />


### Simplified Loss function

<img width="806" alt="image" src="https://github.com/user-attachments/assets/f123d664-e6ee-480c-98df-592ae78d2c7b" />


### Gradient Descent for logistic Regression

<img width="804" alt="image" src="https://github.com/user-attachments/assets/55603e30-8a4c-4f3c-bb1e-517e5e704635" />


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

<img width="804" alt="image" src="https://github.com/user-attachments/assets/27ac5db4-3187-4e9e-bdd9-d2c1c14036b5" />


### The intution behind modified cost function with regularization

- reducing the value of w3 and w4 weights

<img width="804" alt="image" src="https://github.com/user-attachments/assets/73c26dab-5ada-4ddd-83f6-34b662619128" />


- at first we don't know which features are important,so at the beginning we penalize all the 
  weights a little

- lamda balances the goal of fitting data and keeping wj small,setting a proper lamba 
  value is important for addressing the issue of overfitting and underfitting

- increasing lamda will decreases the value of weights

<img width="804" alt="image" src="https://github.com/user-attachments/assets/1e9af2b4-a4b7-4844-a07e-65f662a7809b" />


<img width="804" alt="image" src="https://github.com/user-attachments/assets/84d91a2b-153c-4924-8f48-9f530c277615" />



### Regularized Linear Regression

<img width="804" alt="image" src="https://github.com/user-attachments/assets/bfdefb08-f222-4aee-8c5b-f3337f95f023" />


### Regularized Logistic Regression

<img width="804" alt="image" src="https://github.com/user-attachments/assets/5f60c1a1-8c4a-4603-ae13-d47ac7223ea8" />


<img width="804" alt="image" src="https://github.com/user-attachments/assets/fe07ca20-66e6-472b-9427-6dc19f37b2f8" />


### Why to use DT or not

<img width="804" alt="image" src="https://github.com/user-attachments/assets/a57262fc-505b-4b96-a386-bbd0625737c8" />






