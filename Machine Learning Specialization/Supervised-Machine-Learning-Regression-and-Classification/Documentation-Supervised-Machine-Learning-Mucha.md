
# Week1

## Machine Learning

Machine learning is a field of artificial intelligence that teaches computers to learn from data without being explicitly programmed.

## Types of Machine Learning

- **Supervised Learning**
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**

---

## Supervised Learning

In supervised learning, the algorithm learns from labeled data, meaning each training example has an input and a known correct output (label). The model makes predictions and adjusts based on the accuracy of those predictions.

### Types of Supervised Learning Tasks

| Task Type      | Output Type          | Example                             |
|----------------|----------------------|-------------------------------------|
| Regression     | Continuous Value     | Predicting house prices, stock value |
| Classification | Discrete Class Label | Spam detection                      |


---

## Unsupervised Learning

**Unsupervised Learning** deals with **unlabeled data**. The model explores the underlying structure or patterns in the data without predefined labels.

### Main Applications

- **Clustering** – Grouping similar data points (e.g., customer segmentation)
- **Dimensionality Reduction** – Reducing the number of features (e.g., PCA)
- **Anomaly Detection** – Identifying outliers or unusual data patterns

![image](https://github.com/user-attachments/assets/b214869e-a929-416d-8e84-317ae093a17c)


### Linear Regression Model

Linear Regression is a supervised learning algorithm used to predict a **continuous output variable** based on one or more input features.  
It models the relationship between the input variables (**X**) and the output variable (**y**) using a **straight line**.

![image](https://github.com/user-attachments/assets/0e09ee51-7e61-4117-97db-38e7f48e9e2f) 
![image](https://github.com/user-attachments/assets/81cf56e3-115e-4adc-bb98-1adbb988f334)




### Cost Function

The **cost function** calculates the difference between the predicted values and the actual values from the dataset.

- It quantifies the error in prediction.
**Goal:** Minimize the cost function to improve model accuracy.

  ![image](https://github.com/user-attachments/assets/621f1307-9a16-49bc-8842-6afe7af8c599)


### Gradient Descent

**Gradient Descent** is an optimization algorithm used to minimize the cost function (or loss function) in machine learning models.

- It works by **iteratively updating the model parameters** in the direction that reduces the cost.
- Helps the model learn optimal values for better predictions.
![image](https://github.com/user-attachments/assets/6c49fb31-75e6-44f9-974b-8e7da2aeaf3a)
![image](https://github.com/user-attachments/assets/2caf6571-8b78-46cd-9b57-a7404500129f)

#### Gradient descent for linear regression
![image](https://github.com/user-attachments/assets/6c90aa0b-02fa-418a-b9e0-caeb5f0cccda)


### Learning Rate

The **learning rate**, often denoted by **α (alpha)**, is a **hyperparameter** that determines the size of the steps taken during gradient descent when updating model parameters.

- A **small α** results in slow learning and might take longer to converge.
- A **large α** might overshoot the optimal value or even diverge.

  ![image](https://github.com/user-attachments/assets/53d4a954-4656-40f7-89d3-f9f620e54548)


### Multiple Linear Regression (MLR)

Multiple Linear Regression is a supervised learning algorithm used for predicting a **continuous dependent variable** based on two or more independent variables.

#### Equation:
y = β₀ + β₁x₁ + β₂x₂ + ⋯ + βₙxₙ + ϵ

Where:

- **y**: Dependent variable (target)  
- **x₁, x₂, ..., xₙ**: Independent variables (features)  
- **β₀**: Intercept  
- **β₁, β₂, ..., βₙ**: Coefficients  
- **ϵ**: Error term  
  ![image](https://github.com/user-attachments/assets/e57ee0f4-dfee-4ffb-a07a-87df383957a6)

# Week 2

### Vectorization

**Vectorization** is a key optimization technique in machine learning that involves rewriting algorithms to use **vector and matrix operations** instead of explicit loops.

- It leverages libraries like **NumPy**, which offer low-level, highly optimized implementations.
- **Benefits:**
  - Improved performance
  - Efficient memory usage
  - Hardware acceleration (e.g., SIMD, BLAS libraries)

Vectorized code often runs **significantly faster**, especially on large datasets, making it essential for scalable machine learning workflows.
![image](https://github.com/user-attachments/assets/9c0cbcd2-3d06-4b87-a74f-35bfcd0116bf)


### Vectorization in Gradient Descent

**Vectorization in gradient descent** is a powerful technique that significantly speeds up the optimization process by eliminating explicit loops.

- In traditional gradient descent, weights are updated iteratively using derivatives for each parameter.
- With vectorization, operations are expressed using matrix algebra.
![image-2](https://github.com/user-attachments/assets/52635a01-5eb7-415d-bee2-24321ddf05e8)


###  Gradient Descent for Multiple Linear Regression

Gradient descent is an optimization technique used to train a multiple linear regression model by adjusting the model’s parameters to reduce the error between predicted and actual values. In multiple linear regression, the model tries to find the best-fit line that explains the relationship between several input features and a continuous target variable. Gradient descent works by repeatedly updating the model’s weights in small steps, moving in the direction that reduces the prediction error the most.

![image-3](https://github.com/user-attachments/assets/212bf4db-601f-46fe-bf29-3f98ebb8b283)
![image-4](https://github.com/user-attachments/assets/2ab94617-7335-4bec-be3b-a20964bfc220)

### Feature Scaling

**Feature scaling** is an important technique that can significantly improve the performance of **gradient descent** in machine learning.

![image-8](https://github.com/user-attachments/assets/192f6b4a-1ea4-44d6-93ee-f9bf0a20b2ba)


#### Why It Matters:

When input features have **very different value ranges**, it can lead to inefficient learning during optimization.

##### Example:
- **House size**: 300 to 2000 square feet
- **Number of bedrooms**: 0 to 5

This imbalance causes:
- Features with **larger ranges** to have **smaller weights**
- Features with **smaller ranges** to have **larger weights**

#### Problem:
- The **cost function** becomes elongated and skewed.
- **Gradient descent** takes inefficient zigzag paths.
- Slower convergence.

#### Solution:
Scale features to have **similar ranges**, such as between **0 and 1** (Min-Max Scaling) or **mean 0 and variance 1** (Standardization).

---

#### Common Feature Scaling Methods:

1. **Min-Max Scaling**  
   - Formula: `x_scaled = x / max(x)`
   - Scales features between 0 and 1.
   - Useful when the data is already positive.

2. **Mean Normalization**  
   - Formula: `x_scaled = (x - mean(x)) / (max(x) - min(x))`
   - Centers the data around zero.
![image-6](https://github.com/user-attachments/assets/cc6a542a-d0f8-488b-82c9-5002a0bb326b)

3. **Z-score Normalization (Standardization)**  
   - Formula: `x_scaled = (x - mean(x)) / std(x)`
   - Produces a distribution with mean 0 and standard deviation 1.
   - Works well for many ML algorithms like logistic regression and SVM.
![image-7](https://github.com/user-attachments/assets/9a5cb0a9-6303-4585-86f0-f311935ff6cb)

#### Final Note:

While slight variations in feature ranges are acceptable, features with **extremely large or small values** can slow down training.  
**Rescaling is a simple but powerful preprocessing step** that helps models learn faster and perform better.

![image-5](https://github.com/user-attachments/assets/52ee5e7f-8055-47ab-bc56-94d72aab5276)


### Checking Gradient Descent Convergence

To check if gradient descent is converging, you can plot the cost function \( J(w, b) \) against the number of iterations. A properly converging gradient descent will show a consistently decreasing cost that eventually levels off. If the cost increases or fluctuates significantly, it may indicate that the learning rate is too high or that there's a bug in the implementation. Convergence can be assumed when the change in cost between iterations becomes very small—typically when it's less than a threshold like \( \varepsilon = 0.001 \).

![image-9](https://github.com/user-attachments/assets/bae7f257-e07e-442e-af9e-b6d5d2f8d0c0)


### Choosing the learning rate

To choose a good learning rate, start with a small value like 0.001 and gradually increase it to test how the cost function behaves. If the cost fluctuates or increases, it means the learning rate is too large. Try different values (e.g., 0.001, 0.003, 0.01) and observe the cost at each iteration. Select the largest learning rate that consistently reduces the cost without causing instability.
![image-10](https://github.com/user-attachments/assets/e3a75126-7495-4950-b692-105b12d8e1d6)



### Feature Engineering

Feature engineering is the process of creating new, more informative features to improve the performance of your model. For example, instead of using separate features like width and depth of a house lot, you can create a new feature like the area of the lot (width * depth) that might be more predictive of the house price. By combining or transforming existing features, you can help the algorithm make better predictions. This is a key step in improving your model's effectiveness.

![image-11](https://github.com/user-attachments/assets/83385d93-91ee-4685-a87b-83e48d132f8f)


### Polynomial Regression

Polynomial regression is an extension of linear regression that allows you to fit curves to data by using higher powers of the features. For example, you might use a quadratic or cubic function to model non-linear relationships, such as predicting house prices based on size. When using polynomial features, it's important to scale them to avoid issues with gradient descent, as the powers of the features can drastically increase their range. Feature engineering, like polynomial regression, gives you more flexibility in fitting models that better match your data's patterns.

![image-12](https://github.com/user-attachments/assets/c3cd2a83-c79b-48bf-bebe-15d23304280d)


# Week 3

### Logistic Regression

Logistic regression is a classification algorithm that predicts the probability of a binary outcome (like a tumor being malignant or benign). It uses the sigmoid function to map a linear combination of input features to a value between 0 and 1. The formula is:

f(x) = 1 / (1 + e^-(w·x + b))

The output is interpreted as the probability of the positive class (label 1), and if it's greater than 0.5, class 1 is predicted.
![image-13](https://github.com/user-attachments/assets/03b60c6f-7565-4a55-9387-3e38cdaf7c6d)


### Decision Boundary

The decision boundary is defined where `z = 0`; for linear models, this forms a straight line (e.g., `x₁ + x₂ = 3`), and for polynomial features, it can become more complex shapes like circles or ellipses. By adding higher-order polynomial features, logistic regression can model more complex data patterns. The next step is to learn the cost function and use gradient descent for training.
![image-14](https://github.com/user-attachments/assets/1be2456b-47df-4458-a8da-1ab6332ad066)
![image-15](https://github.com/user-attachments/assets/8d1e5f40-09b7-41df-8702-460cacaa6cd8)

## Cost function for logistic regression

The squared error cost function is unsuitable for logistic regression due to its non-convex nature, leading to multiple local minima during gradient descent. A new cost function based on log loss is introduced, ensuring a convex cost surface and enabling gradient descent to converge to the global minimum. This new cost function improves the logistic regression model's prediction accuracy.
![image-16](https://github.com/user-attachments/assets/0efc8422-df00-4dd9-9727-a07df757bf4d)


### Simplified Cost Function for Logistic Regression

It explains how to simplify the loss and cost functions for logistic regression, making gradient descent easier to implement. The loss function is written as a single equation for both possible outcomes (0 or 1), removing the need for separate cases. The cost function, which averages the losses, is convex, ensuring gradient descent can effectively find the optimal parameters. This cost function is derived from maximum likelihood estimation, which helps efficiently estimate model parameters.
![image-17](https://github.com/user-attachments/assets/63b8de12-efe2-4455-b5d0-cf5135838b64)


### Gradient Descent  for Logistic Regression

It explains how to use gradient descent to minimize the cost function in logistic regression by updating parameters based on their derivatives. It also highlights the importance of feature scaling and introduces a vectorized implementation to speed up the process.
![image-18](https://github.com/user-attachments/assets/0bb660d1-4a9c-43fe-86f6-f7cdbe7c52b3)


### The problem of overfitting
Overfitting in machine learning occurs when a model learns the training data too well, including its noise and random fluctuations, leading to poor performance on new, unseen data.
![image-19](https://github.com/user-attachments/assets/4bb60328-a1b7-4f16-a44b-a512b4babf76)
![image-20](https://github.com/user-attachments/assets/a11dd42f-bd5e-4330-9e88-a56950e42b8a)


### Addressing overfitting
This part strategies for addressing overfitting in machine learning models. When a model is overfit, one way to reduce overfitting is to gather more training data, which helps the model generalize better. However, if more data isn't available, another approach is feature selection, where you choose only the most relevant features to use for training, reducing the complexity of the model. Lastly, regularization is introduced as a technique to reduce overfitting by shrinking the values of the model parameters, without necessarily eliminating features. Regularization helps ensure that no individual feature has too large an impact on the model.

![image-21](https://github.com/user-attachments/assets/f99e24e7-2a78-43fa-b55a-4ac3403e9f8a)


### Cost function with regularization

Regularization is a technique used to prevent overfitting in machine learning models by discouraging overly complex models. The core idea is to modify the cost function to penalize large parameter values, effectively encouraging smaller weights. This leads to simpler, smoother models that generalize better to new data. The strength of regularization is controlled by a parameter called lambda. If lambda is set to zero, the model ignores regularization and may overfit the training data. On the other hand, if lambda is extremely large, the model becomes too simple (like a flat line) and underfits. The goal is to find a balanced lambda value that keeps weights small without sacrificing the model’s ability to fit the data well. By default, regularization is applied to all weights (but usually not the bias term) because we often don’t know in advance which features are most important. This technique is widely used in both linear and logistic regression, and it helps create models that perform better on unseen data by avoiding extreme complexity. Choosing the right lambda is crucial and is typically done through model selection techniques.
![image-22](https://github.com/user-attachments/assets/e27485cf-ee81-4b44-9428-3bf47debaa6e)
![image-23](https://github.com/user-attachments/assets/25efd7f3-6708-4a65-a9af-2ae86e461216)

### Regularized linear regression

Regularization modifies gradient descent by adding a weight-shrinking effect to prevent overfitting. For the bias term (b), updates remain unchanged. For weights (w_j), each update now includes two parts: 1) the original gradient step from the cost function, and 2) an extra term that slightly shrinks the weight toward zero. This shrinkage happens because regularization adds a small penalty proportional to the weight's current value during each update. The strength of this effect depends on the regularization parameter - larger values cause stronger shrinkage. The learning rate and training set size also influence how much shrinking occurs. This process automatically keeps weights small while still fitting the data, resulting in simpler models that generalize better. The same approach can be extended to logistic regression.
![image-24](https://github.com/user-attachments/assets/b26e3469-17e7-4490-a292-4c2da14ecae4)

### Regularized logistic regression

Regularized logistic regression helps prevent overfitting, especially when using complex features like high-order polynomials, by gently pushing the model toward simpler solutions. Just like with linear regression, we tweak the training process to shrink unnecessary weights without touching the bias term. The strength of this smoothing effect is controlled by a tuning parameter: higher values create smoother decision boundaries that generalize better to new data, while lower values allow more complex fits. The implementation works almost identically to standard logistic regression, with one key difference—each weight update now includes a small nudge to keep weights modest in size. This elegant solution lets logistic regression handle many features while avoiding overly wiggly decision boundaries. These same principles of regularization and gradient descent form the foundation for more advanced techniques like neural networks, showing how core concepts scale up to cutting-edge AI. Mastering this balance between fitting and simplicity is what separates practical ML applications from academic exercises.
![image-25](https://github.com/user-attachments/assets/ba28627f-6639-4f18-9027-6fdab8e45b78)
