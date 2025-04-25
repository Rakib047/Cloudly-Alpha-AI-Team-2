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

# Gradient Descent

Gradient Descent helps to minimize the cost function more systematically. 

![gradient_descent_1](https://github.com/user-attachments/assets/10c68b66-4d7c-4642-bddc-99cfaf13b2dd)

- **Gradient Descent Update Rule**  

  - `w ← w - α * ∂J/∂w(w,b)`  
  - `b ← b - α * ∂J/∂b(w,b)`  
  - Each step adjusts parameters by moving “downhill” on the cost surface.



- **Learning Rate (α)**  
  - A small positive scalar (e.g. `0.01`).  
  - Controls step size: large α → big jumps; small α → small, calculative steps.

- **Derivative Term (∂J/∂w, ∂J/∂b)**  
  - Tells direction to reach the local minimum.


- **Two-Parameter Model**  
  - Linear regression has both `w` (slope) and `b` (intercept).  
  - Both parameters get updated each iteration.

- **Simultaneous vs. Sequential Updates**  
  - **Correct (simultaneous):**  
    ```python
    temp_w = w - α * dJ_dw(w, b)
    temp_b = b - α * dJ_db(w, b)
    w = temp_w
    b = temp_b
    ```
  - **Incorrect (sequential):**  
    ```python
    temp_w = w - α * dJ_dw(w, b)
    w = temp_w
    temp_b = b - α * dJ_db(w, b)  # uses updated w
    b = temp_b
    ```
  - Simultaneous update ensures both derivatives use the same “old” values.

- **Convergence**  
  - Repeat updates until `w` and `b` change very little (local minimum reached).

# Gradient Descent Intuition

- **Gradient Descent (one-parameter version)**  
  - Update rule:  
    ```text
    w ← w - α * (d/dw J(w))
    ```
  - Goal: adjust `w` to minimize cost $J(w)$.

- **Tangent Line & Derivative**  
  - At any `w`, draw the tangent line to $J(w)$.  
  - Slope of tangent = derivative $\frac{d}{dw}J(w)$.  
  - Positive slope → tangent rises rightward; negative slope → falls.

- **Case 1: Starting on the Right (Positive Slope)**  
  - $\frac{d}{dw}J(w) > 0$
  - Update: $w_{\text{new}} = w - \alpha \times (\text{positive})$ → $w_{\text{new}} < w$ 
  - Moves `w` left, toward the minimum.

- **Case 2: Starting on the Left (Negative Slope)**  
  - $\frac{d}{dw}J(w) < 0$  
  - Update: $w_{\text{new}} = w - \alpha \times (\text{negative})$ → $w_{\text{new}} > w$ 
  - Moves `w` right, toward the minimum.

- **Why It Works**  
  - Derivative gives direction:  
    - Positive → decrease `w` 
    - Negative → increase `w` 
  - Always seeks to reduce cost.

- **Learning Rate ($\alpha$)**  
  - Scales step size.  
  - Too small → very slow convergence.  
  - Too large → may overshoot or diverge.  

# Learning Rate ($\alpha$)

- **Gradient Descent**  
  $
  w \leftarrow w - \alpha \,\frac{d}{dw}J(w)
  $  
  $\alpha$ scales the derivative step.

- **Too Small $\alpha$**  
  - $\alpha \ll 1$ (e.g. $10^{-7}$)  
  - Each update: tiny small step 
  - Convergence: very slow, requires many iterations  
  - Outcome: cost $J(w)$ decreases but inefficiently

- **Too Large $\alpha$**  

  - Each update: huge jump  
  - May overshoot minimum → cost increases  
  - Can oscillate or diverge (never settle)

- **Behavior at a Local Minimum**  
  - At minimum: $\frac{d}{dw}J(w)=0$ 
  - Update:  
    $
    w_{\text{new}} = w - \alpha \times 0 = w
    $ 
  - Parameters remain unchanged (desired)

- **Automatic Step-Size Decay**  
  - As `w` approaches minimum, $\left|\tfrac{d}{dw}J(w)\right|\to0$  
  - Updates shrink, even with fixed $\alpha$ 
  - Ensures finer convergence near optimum

# Linear Regression with Gradient Descent

- **Model & Cost Function**  
  - Linear model: $f(x) = w x + b$ 
  - Squared error cost:  
    $
      J(w,b) = \frac{1}{2m}\sum_{i=1}^{m} \bigl(f(x^{(i)}) - y^{(i)}\bigr)^2
    $

- **Gradient Descent Updates**  
  $
    \begin{aligned}
      w &\leftarrow w - \alpha \,\frac{\partial}{\partial w}\,J(w,b) \\
      b &\leftarrow b - \alpha \,\frac{\partial}{\partial b}\,J(w,b)
    \end{aligned}
  $

- **Calculated Derivatives**  
  - With respect to `w`:  
    $
      \frac{\partial}{\partial w}\,J(w,b)
      = \frac{1}{m}\sum_{i=1}^{m}\Bigl(f(x^{(i)}) - y^{(i)}\Bigr)\,x^{(i)}
    $
  - With respect to `b`:  
    $
      \frac{\partial}{\partial b}\,J(w,b)
      = \frac{1}{m}\sum_{i=1}^{m}\Bigl(f(x^{(i)}) - y^{(i)}\Bigr)
    $

- **$\tfrac{1}{2m}$ Factor**  
  - Derivative of $\tfrac{1}{2m}(error)^2$ brings down a 2 that cancels the $\tfrac12$, simplifying the update formulas.

- **Convexity & Convergence**  
  - Squared error cost for linear regression is a **convex** “bowl-shaped” function.  
  - No local minima: only one global minimum.  
  - Properly chosen $\alpha$ guarantees convergence to the global optimum.


# Running Gradient Descent

- **Initialization**  
  - Start with `w = -0.1`, `b = 900` → `f(x) = -0.1x + 900`.  


- **First Update Step**  
  - Gradient descent moves cost from initial point down and to the right on contour plot.  
  - Model line shifts slightly toward the data.

- **Subsequent Steps & Trajectory**  
  - Each step reduces cost; parameter pair `(w,b)` traces a path to the global minimum.  
  - Corresponding line fit steadily improves.

- **Final Fit & Prediction**  
  - At convergence, model line is the best straight-line fit to the data.  
  - Example: for `x`=1250, read off predicted $y\approx\$250{,}000$.

- **Batch Gradient Descent**  
  - “Batch” ⇒ each update uses **all** training examples (sum over $i=1\ldots m$).  
  - Contrast: other variants use subsets (mini‑batch or stochastic).

## Multiple Linear Regression Model

- **General Model**
  - $ f_{w,b}(x) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b $

- **Concrete Example**
  - $f_{w,b}(x) = 0.1x_1 + 4x_2 + 10x_3 - 2x_4 + 80 $
  - assuming output in $1000s:
    - Base price = $80,000  
    - +$100 per extra square foot  
    - +$4,000 per bedroom  
    - +$10,000 per floor  
    - -$2,000 per year of age

# Vectorization 

- **Purpose**
  - Makes code shorter and cleaner.
  - Increases computational efficiency using optimized libraries and hardware (like CPUs and GPUs).

---



## Definition 
  - Writing operations on entire vectors or matrices instead of using loops.
  - Enables use of efficient low-level implementations provided by numerical libraries.

- **Example**
  - Given:
    - $ \vec{w} = [w_1, w_2, w_3] $
    - $ \vec{x} = [x_1, x_2, x_3] $
    - $ b $: bias term
  - \( n = 3 \), meaning there are 3 features.

---

## Without Vectorization

- **Manual Computation**
  ```python
  f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b


# Vectorization Part 2


## Pros of Vectorization


  - Vectorization dramatically speeds up code execution.


---

##  Sequential vs Parallel Execution

- **Without Vectorization (For Loop)**
  - Executes operations **sequentially**, one after another.
  - Example: For loop from `j = 0` to `j = 15` performs one multiplication and addition per iteration.
  - Takes **multiple time-steps** to complete (t₀ to t₁₅).

- **With Vectorization (NumPy functions)**
  - Performs all operations in **parallel** using optimized hardware (like CPU SIMD or GPU cores).
  - Multiples all elements of `w` and `x` together in one step.
  - Then **adds all results** using optimized routines—much faster than individual additions.
  - Greatly reduces computation time, especially for large vectors.

---


## Machine Learning Use Case
  - Vectorization is essential for training on **large datasets** and **complex models**.
  - Allows ML algorithms to **scale efficiently**.
  - Especially critical for deep learning and big data.

---

## Example: Gradient Descent in Linear Regression

- **Scenario**
  - 16 features and 16 corresponding weights $ w_1 $ to $ w_{16} $, with derivatives $ d_1 $ to $ d_{16} $.
  - Learning rate: 0.1

- **Non-Vectorized Update**
  ```python
  for j in range(16):
      w[j] = w[j] - 0.1 * d[j]

- **Vectorized Update**
  ```python
  w = w - 0.1*d

# Gradient Descent for Multiple Linear Regression with Vectorization

---

##  Multiple Linear Regression with Vectors

- **Parameter Vector**
  - Combine all weights $ w_1, w_2, \ldots, w_n $ into a single vector $ \mathbf{w} $.
  - Bias term $ b $ remains a separate scalar.

- **Model Hypothesis**
  - Written compactly as:  
    $ f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b $  
  - The dot (·) denotes dot product between $ \mathbf{w} $ and input vector $ \mathbf{x} $.

- **Cost Function**
  - Previously: $ J(w_1, w_2, \ldots, w_n, b) $  
  - Now: $ J(\mathbf{w}, b) $, taking vector $ \mathbf{w} $ and scalar $ b $ as inputs.




---

## Gradient Descent in Vector Form

- **Update Rule Overview**
  - Each parameter $ w_j $ is updated using:  
    $ w_j := w_j - \alpha \cdot \frac{\partial J}{\partial w_j} $
  - $ b $ is updated with its own derivative.

- **From Univariate to Multivariate**
  - **Univariate Regression**:
    - Only one weight $ w $ and one input $ x $.
    - Gradient update:  
      $ w := w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m}(f(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $

  - **Multivariate Regression**:
    - Update rule is generalized for all $ n $ features.
    - For each feature $ j $:  
      $ w_j := w_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m}(f(\mathbf{x}^{(i)}) - y^{(i)}) \cdot x_j^{(i)} $ 

- **Vectorized Implementation**
  - Vectorized form updates **all weights at once**:  
    ```python
    w = w - alpha * dw
    b = b - alpha * db
    ```
  - Where `dw` and `db` are the gradients w.r.t. `w` and `b`.

---

## Normal Equation: An Alternative to Gradient Descent

- **Defination**
  - A **closed-form** solution to linear regression.
  - Uses linear algebra to solve for $ \mathbf{w} $ and $ b $ without iteration.

- **Advantages**
  - Direct computation of optimal values.
  - No need for learning rate or iteration.

- **Disadvantages**
  - Works **only for linear regression**.
  - **Does not generalize** to other ML algorithms (e.g., logistic regression, neural networks).
  - Can be **slow for high-dimensional data** (many features).

- **Purpose**
  - Rarely implemented by hand.
  - Some ML libraries may use it internally for efficiency.


---

# Feature Scaling 



- **Different Feature Ranges**
  - Features can vary greatly in range.
  - Example:
    - `x1`: House size in square feet (300 - 2000)
    - `x2`: Number of bedrooms (0 - 5)
  - Uneven feature scales can make learning inefficient.

- **Impact on Parameters**
  - Large feature ranges → Smaller parameter values (e.g., `w1 = 0.1`)
  - Small feature ranges → Larger parameter values (e.g., `w2 = 50`)
  - Parameters adjust inversely to feature scale to maintain balanced predictions.

---

## Gradient Descent Challenges with Unscaled Features

- **Contour Plot Behavior**
  - With unscaled features, cost function contours appear elongated (elliptical).
  - One axis (e.g., `w1`) affects the cost much more than the other.
  - Gradient descent may:
    - Zigzag back and forth
    - Take longer to converge

- **Example Scenario**
  - For input: 2000 sq ft and 5 bedrooms
  - Poor parameters (e.g., `w1 = 50, w2 = 0.1`) → Overestimated price
  - Better parameters (e.g., `w1 = 0.1, w2 = 50`) → Accurate prediction

---

## Feature Scaling Solution

- **Defination**
  - Transform input features to similar ranges.
  - Example: Scale both `x1` and `x2` to range from 0 to 1.

- **Effect of Scaling**
  - Transforms cost function contours into more circular shapes.
  - Gradient descent follows a more direct path to the minimum.
  - **Significantly improves convergence speed.**

---


- **Problem**
  - Features with very different value ranges slow down gradient descent.

- **Solution**
  - Rescale features to have **comparable ranges**.

- **Result**
  - Gradient descent becomes faster and more efficient.
  - Enables better and quicker convergence to the optimal parameters.


---
# Feature Scaling Techniques in Gradient Descent

---

## Purpose of Feature Scaling

- Ensures all features take **comparable ranges** of values.
- Improves the **efficiency and convergence** of gradient descent.
- Prevents some parameters from dominating due to large feature scales.

---

## 1. **Simple Rescaling (Min-Max Normalization)**

- **Approach**: Divide each feature by its maximum value.
  - `x1_scaled = x1 / 2000` → Now ranges from ~0.15 to 1.
  - `x2_scaled = x2 / 5` → Now ranges from 0 to 1.
- **Effect**: Brings features into a similar range (e.g., 0 to 1).

---

## 2. **Mean Normalization**

- **Approach**: Center features around 0.
  - Formula:  
    `x1_norm = (x1 - μ1) / (max1 - min1)`  
    `x2_norm = (x2 - μ2) / (max2 - min2)`
- **Example**:
  - If μ1 = 600, max1 = 2000, min1 = 300 → `x1_norm` ranges from ~-0.18 to 0.82.
  - If μ2 = 2.3, max2 = 5, min2 = 0 → `x2_norm` ranges from ~-0.46 to 0.54.
- **Effect**: Centers data around zero with a comparable scale.

---

## 3. **Z-Score Normalization (Standardization)**

- **Approach**: Normalize using **mean** and **standard deviation**.
  - Formula:  
    `x_norm = (x - μ) / σ`
- **Example**:
  - For `x1`: μ1 = 600, σ1 = 450 → `x1` ranges ~-0.67 to 3.1.
  - For `x2`: μ2 = 2.3, σ2 = 1.4 → `x2` ranges ~-1.6 to 1.9.
- **Effect**: Commonly used method that transforms data to have zero mean and unit variance.

---

## 4. **What Ranges to Aim For**

- Target range: roughly `-1 to +1`
  - Looser bounds like `-3 to +3` or `-0.3 to +0.3` are also acceptable.
- It's **not mandatory** to rescale if:
  - Feature values are already in a narrow or consistent range.

---

## 5. **When to Rescale**

- **Must scale** if:
  - Feature range is **too large** (e.g., `x3: -100 to +100`).
  - Feature values are **too small** (e.g., `x4: -0.001 to 0.001`).
  - Feature values are **large constants** (e.g., `x5: 98.6 to 105` for temperature).
- **General rule**: If unsure, just rescale. It won’t hurt and likely helps.

---

## Pointers

- **Feature scaling** helps gradient descent **converge faster** and more reliably.
- Use **Min-Max Scaling**, **Mean Normalization**, or **Z-score Normalization**.
- Always consider scaling if features are on **vastly different scales**.
- Prepares the data well for algorithms sensitive to feature magnitude.


---
# Gradient Descent Convergence

---

## Purpose

- To verify if gradient descent is **converging properly**, i.e., minimizing the cost function $ J $ over time.
- Helps detect problems like a **poor learning rate** or bugs in the implementation.

---

## Plotting the Learning Curve

- **What it is**: A plot of cost function $ J(w, b) $ vs. number of iterations.
  - **X-axis**: Number of iterations (after each update of $ w $ and $ b $).
  - **Y-axis**: Cost $ J(w, b) $ computed using current parameters.
- **Insight**: This shows how the cost evolves over time, not how it changes with respect to the parameter values.

---

## What a Good Learning Curve Looks Like

- **Consistent Decrease**: If gradient descent is implemented and configured properly, cost $ J $ should **decrease after every iteration**.
- **Flattening Out**: When the curve **levels off**, gradient descent has likely **converged**.
  - E.g., if cost stops decreasing significantly after 300-400 iterations.

---

## Indication of faltering

- **Cost Increases**: Indicates a **bad learning rate (too large)** or a **bug in the code**.
- **No Change or Flatline** early on: Possibly too **small of a learning rate** or **stuck in a poor region**.

---

## When Does Gradient Descent Converge?

- **Highly variable**: 
  - Some applications: convergence in ~30 iterations.
  - Others: might take **1,000 to 100,000** iterations.
- It's hard to know in advance — hence, plotting is key.

---

## Automatic Convergence Check

- **Use of Epsilon $ \varepsilon $**:
  - Let $ \varepsilon $ be a **small number** like 0.001 or $ 10^{-3} $.
  - If the cost decreases by **less than $ \varepsilon $** over an iteration, declare convergence.
- **Practical Note**: Choosing $ \varepsilon $ can be tricky.
  - Visual inspection of the learning curve is **usually more reliable** than automated checks.

---

## Pointers

- Always plot the **cost vs. iterations** to observe learning behavior.
- A **well-behaved learning curve** decreases smoothly and levels off.
- Look out for **jumps, flatlines, or inconsistencies** in the curve.
- Use **epsilon-based convergence** only as a supplement to visual checks.



---
# Choosing a Good Learning Rate (Alpha)



## Why Learning Rate Matters

- A **proper learning rate (Alpha)** is crucial for efficient training.
  - **Too small**: Gradient descent runs **very slowly**.
  - **Too large**: May **not converge** and cost may oscillate or increase.

---

## Signs of a Bad Learning Rate

- **Cost oscillates (up and down)**:
  - Could be due to a **too-large learning rate**.
  - May also indicate a **bug in the code**.

- **Cost increases steadily**:
  - Likely a **too-large learning rate**.
  - Or an **incorrect update rule**, e.g., updating with:
    ```
    w1 = w1 + Alpha * dJ/dw1  ❌
    ```
    instead of:
    ```
    w1 = w1 - Alpha * dJ/dw1  ✅
    ```

---

## Explanation of Overshooting

- With **large Alpha**, each step may **overshoot** the minimum and bounce back and forth.
- Results in **non-decreasing** or unstable cost.
- A **smaller Alpha** makes more **stable and consistent** descent.

---

## Debugging Tip

- Set **Alpha to a very small value** and see if **cost decreases every iteration**.
  - If not, this suggests a **bug** in your code.
- Note: This is just for debugging — such a small Alpha is not practical for training.

---

## Trade-off with Small Learning Rates

- **Very small Alpha** ensures consistent decrease in cost but:
  - Requires **more iterations**.
  - Slows down training.

---

## Strategy for Choosing Learning Rate

- Try **multiple values** for Alpha:
  - Start from **0.001**
  - Multiply by ~3x each step:
    - Try: 0.001 → 0.003 → 0.01 → 0.03 → 0.1, etc.
- **For each Alpha**, run gradient descent for **a few iterations** and **plot cost**.

- Pick the Alpha that:
  - **Decreases cost consistently**
  - Does so **as rapidly as possible** without instability

- Ensure you test:
  - A value that is **too small**
  - A value that is **too large**
  - Choose something **slightly smaller** than the largest stable Alpha

---

## Pointers

- Plotting **cost vs. iterations** for each Alpha helps in picking the best learning rate.
- This visual feedback helps you avoid blindly choosing hyperparameters.

---

# Feature Engineering


## Importance of Feature Choice

- The **choice of features** can greatly affect a learning algorithm's **performance**.
- In many real-world applications, **good feature selection/engineering** is crucial for success.

---

## Revisiting the House Price Prediction Example

- **Original features**:
  - `x₁`: Frontage (width) of the lot
  - `x₂`: Depth of the lot
- Model using original features:
  ```
  f(x) = w₁x₁ + w₂x₂ + b
  ```
  - Works, but may not capture the most informative pattern.

---

## Creating a New Feature (Feature Engineering)

- You can define a **new feature**:
  - `x₃ = x₁ * x₂` → area of the lot
- Intuition: **Area** might be more predictive of house price than frontage and depth individually.
- New model using engineered feature:
  ```
  f(x) = w₁x₁ + w₂x₂ + w₃x₃ + b
  ```
  - Allows the model to **learn** which features are most useful.

---

## Feature Engineering

- **Definition**: The process of creating new features using **domain knowledge** and **intuition**.
- Often involves:
  - **Transforming** original features
  - **Combining** multiple features
- Purpose: Make it **easier** for the learning algorithm to **learn accurate patterns**.

---

## Benefits of Feature Engineering

- Helps the algorithm:
  - Make **better predictions**
  - Capture **important relationships** in the data
- Can lead to a **significantly better model** compared to using raw features only.

---

# Polynomial Regression and Feature Engineering 


- Polynomial regression allows us to fit **curves** and **non-linear functions** to our data.
- Achieved by **adding polynomial terms** (like x², x³) as features.

---

## When to Use Polynomial Regression

- Consider a dataset where the feature is the **size of a house (x)**.
- If data doesn’t fit well with a straight line, try fitting a **quadratic function** (x²).
- But a **quadratic function** eventually comes back down — not always ideal (e.g., housing prices shouldn't decrease with increasing size).
- A **cubic function** (x³) might be better because it keeps increasing, better matching the trend of increasing prices with larger homes.

---

## Features in Polynomial Regression

- A **quadratic model** uses:
  - x (original feature)
  - x² (size squared)
- A **cubic model** uses:
  - x
  - x²
  - x³

These are **examples of polynomial regression**.

---

## Importance of Feature Scaling

- Polynomial features can have **very different value ranges**:
  - x: 1–1,000
  - x²: 1–1,000,000
  - x³: 1–1,000,000,000
- These differences can **negatively impact gradient descent** performance.
- **Feature scaling** is essential for making sure all features have similar ranges.

---

## Alternative Feature Choices

- Instead of polynomial powers, consider **non-linear transformations** like:
  - √x (square root of x)
- Example model:
  - f(x) = w₁·x + w₂·√x + b
- The **square root function** increases more slowly, doesn't flatten or decrease — might work well in some cases.

---

## Choosing the Right Features

- It’s not always obvious which features to use.
- Later, you’ll learn how to **compare models** that use different features to evaluate performance.
- For now, know that **you have control** over the features used.
- Use **feature engineering** and **domain knowledge** to help build better models.

---

# Classification and Logistic Regression


---

## Classification Problems
- **Classification**: Output variable `y` can only be one of a few possible values.
- **Examples**:
  - Email spam detection (Spam or Not Spam)
  - Fraudulent financial transaction (Fraud or Not Fraud)
  - Tumor classification (Malignant or Benign)
- These are **binary classification problems** (only 2 possible outputs).

---

## Terminology
- Common labels: `No/Yes`, `False/True`, or `0/1`.
- Conventionally, `0` = Negative class, `1` = Positive class.
- "Positive" and "Negative" are not related to good/bad, just presence/absence of a trait.

---



### Problem with Linear Regression
- Adding an **outlier** (e.g., a new point far to the right) shifts the best-fit line.
- This **distorts the decision boundary** and results in incorrect classifications.
- Linear regression is **not robust** for classification tasks.

---

## Logistic Regression Introduction
- Designed for binary classification.
- Output is **always between 0 and 1**.
- Avoids pitfalls of linear regression in classification settings.



---

# Logistic Regression - Introduction & Intuition

- Logistic regression is one of the **most widely used classification algorithms**.
- Used to predict **binary outcomes** (e.g., 0 or 1, malignant or benign).

---

## Classification Setup
- **Label 1 (Yes/Positive)** = Malignant tumor.
- **Label 0 (No/Negative)** = Benign tumor.
- Dataset plots tumor size (x-axis) vs class label (y-axis: 0 or 1).

---

## Logistic Regression vs Linear Regression
- Linear regression fits a **straight line**, which can output values < 0 or > 1 — not ideal for classification.
- Logistic regression fits an **S-shaped curve (Sigmoid)** to predict values between **0 and 1**.

---

## Sigmoid (Logistic) Function

  $
  g(z) = \frac{1}{1 + e^{-z}}
  $
- **Properties**:
  - Output is always between **0 and 1**.
  - If `z = 0`, then `g(z) = 0.5`.
  - As `z → +∞`, `g(z) → 1`.
  - As `z → -∞`, `g(z) → 0`.

---

## Building the Logistic Regression Model
1. Compute linear function:  
   $
   z = w \cdot x + b
   $
2. Apply sigmoid function:  
   $
   f(x) = g(z) = \frac{1}{1 + e^{-z}}
   $

- The model **maps input features `x` to a probability** between 0 and 1.

---

## Interpreting Output
- `f(x)` = predicted **probability that `y = 1`** given input `x`.
- Example: If `f(x) = 0.7`, model predicts a **70% chance** that the tumor is malignant.
- Then, **P(y = 0) = 1 - 0.7 = 0.3 (30%)**.

---

## Mathematical Notation (FYI)
- Sometimes expressed as:  
  $
  f(x) = P(y = 1 \mid x; w, b)
  $
- Semicolon means that `w` and `b` are parameters of

---

# Logistic Regression: Decision Boundary


- **Two-step computation**: 
  - Compute `z = w · x + b`
  - Apply sigmoid: `f(x) = g(z) = 1 / (1 + e^(-z))`
- **Output interpretation**:
  - `f(x)` is interpreted as the **probability** that `y = 1` given `x`.

## Making Predictions
- **Threshold Rule**:
  - If `f(x) ≥ 0.5`, then predict `ŷ = 1`.
  - If `f(x) < 0.5`, then predict `ŷ = 0`.

## When is `f(x) ≥ 0.5`?
- Sigmoid function `g(z)` is ≥ 0.5 when `z ≥ 0`
- Since `z = w · x + b`, prediction is:
  - `ŷ = 1` when `w · x + b ≥ 0`
  - `ŷ = 0` when `w · x + b < 0`

## Decision Boundary
- **Definition**: 
  - The line (or surface) where `w · x + b = 0`
  - Separates the feature space into regions predicting `y = 1` and `y = 0`
- **2D Example with two features (x₁, x₂)**:
  - Parameters: `w₁ = 1`, `w₂ = 1`, `b = -3`
  - Decision boundary equation: `x₁ + x₂ = 3`

## Non-Linear Decision Boundaries
- **Using Polynomial Features**:
  - Set `z = w₁x₁² + w₂x₂² + b`
  - Example: If `w₁ = w₂ = 1` and `b = -1`, then:
    - Decision boundary: `x₁² + x₂² = 1` → a **circle**
    - Outside circle (`z ≥ 0`): predict `y = 1`
    - Inside circle (`z < 0`): predict `y = 0`

## More Complex Boundaries
- **Using higher-order polynomial terms**:
  - Example: `z = w₁x₁ + w₂x₂ + w₃x₁² + w₄x₁x₂ + w₅x₂²`
  - Can create **elliptical or irregular shaped** boundaries
  - Logistic regression can fit **complex data** with appropriate feature transformation

  ---

  # Logistic Regression Cost Function 



- **Cost Function Purpose**  
  Measures how well a set of parameters (weights `w` and bias `b`) fits the training data.

- **Choosing Parameters**  
  Helps in finding better parameters by minimizing the cost function.

---

## Training Set and Model

- **Training Data**  
  Example: Patient diagnosis data (e.g., tumor size, age, etc.)

- **Number of Examples**  
  Denoted by `m`.

- **Number of Features**  
  Denoted by `n` (features = `X_1, X_2, ..., X_n`).

- **Target Label**  
  Binary classification: `y ∈ {0, 1}`.

- **Model Definition**  
  Logistic regression prediction:  
  `f(x) = 1 / (1 + e^(-w·x + b))`

---

## Why Squared Error Doesn't Work

- **Linear Regression Cost Function**
  ```math
  J(w, b) = \frac{1}{2m} \sum (f(x^{(i)}) - y^{(i)})^2

---

# Simplified Loss and Cost Functions for Logistic Regression


- **Simplify** the loss and cost functions for logistic regression to make **gradient descent** implementation easier.
- Use a more efficient form of the **binary classification** loss function.

---

##  Simplified Loss Function
- **Binary Classification**: `y` can only be 0 or 1.
- The loss function is defined as:
  ```math
  L(f(x), y) = -y \cdot \log(f(x)) - (1 - y) \cdot \log(1 - f(x))
  ```
  - `f(x)` = model prediction (sigmoid output)
  - `y` = actual label (0 or 1)

---

##  Why the Simplified Loss Works
- **When `y = 1`**:
  - The first term: `-1 * log(f(x))`
  - The second term: becomes `0` since `1 - y = 0`
- **When `y = 0`**:
  - The first term becomes `0` since `y = 0`
  - The second term: `-1 * log(1 - f(x))`
  
- **Result**: The simplified loss function works for both cases (when `y = 0` or `y = 1`) without needing separate cases.

---

## Cost Function for Logistic Regression
- **Cost Function** is the **average loss** over all training examples:
  ```math
  J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)})
  ```
  where `m` is the number of training examples.

- **Plugging in the simplified loss**:
  ```math
  J(w, b) = \frac{1}{m} \sum_{i=1}^{m} -y^{(i)} \log(f(x^{(i)})) - (1 - y^{(i)}) \log(1 - f(x^{(i)}))
  ```

---

## Why This Cost Function?
- The cost function is derived from **Maximum Likelihood Estimation (MLE)**, a statistical technique for estimating parameters.
- **Convexity**: The cost function is convex, ensuring **efficient optimization** during gradient descent.



---

# Gradient Descent & Parameter Optimization

## Objective
- **Goal**: Minimize the **cost function** `J(w, b)` by finding optimal parameters `w` and `b`.
- Once trained, the model can predict the probability that `y = 1` for a new input `x`.

## Gradient Descent for Logistic Regression
- **Gradient descent** is used to minimize the cost function.
- General update rule for each parameter:
  ```math
  \theta := \theta - \alpha \cdot \frac{\partial J}{\partial \theta}
  ```

## Partial Derivatives
- Derivative of cost `J` with respect to `w_j`:
  ```math
  \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
  ```

- Derivative of cost `J` with respect to `b`:
  ```math
  \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})
  ```

##  Simultaneous Updates
- All parameter updates must be computed first, then applied **simultaneously** to avoid inconsistency during the iteration.

## Clarification: Logistic ≠ Linear Regression
- Even though the update rules look similar to linear regression, they're **different algorithms**:
  - **Linear Regression**:  
    `f(x) = w·x + b`
  - **Logistic Regression**:  
    `f(x) = sigmoid(w·x + b)`

## Vectorization
- Gradient descent can be **vectorized** for faster computation, as done in linear regression.
- Details and implementation examples are available in optional labs.

## Feature Scaling
- Scaling features to a similar range (e.g., [-1, 1]) can **speed up convergence** of gradient descent.
- Works the same way as in linear regression.

---
# Overfitting and Underfitting in Machine Learning



## Overfitting
- **Overfitting** occurs when a model fits the training data **too well**, capturing noise and fluctuations.
- This results in **poor generalization** to new, unseen data.
  
- **Example**: Fitting a **fourth-order polynomial** to predict housing prices.
  - The model passes through every training point perfectly.
  - However, this is **overfit** and will perform poorly on new data.

- **Characteristics of Overfitting**:
  - **High Variance**: The model's predictions vary significantly with small changes in the training data.
  - **Complexity**: The model is overly complex (e.g., high-order polynomials), capturing irrelevant details.

---

## Underfitting
- **Underfitting** occurs when the model is too **simple** to capture the underlying patterns in the data.
- It fails to perform well even on the training set.

- **Example**: Fitting a **linear model** to predict housing prices when the relationship is more complex (e.g., quadratic).
  - This model has **high bias** and cannot capture the true data patterns.

- **Characteristics of Underfitting**:
  - **High Bias**: The model makes incorrect assumptions and doesn't fit the training data well.

---

##  Balancing Overfitting and Underfitting
- **Ideal Model**: A model that **generalizes well**, fitting the training data adequately without overfitting.
  - This model should have **low bias** and **low variance**.

- **Goldilocks Analogy**:
  - Too **simple** (underfitting) → high bias, poor performance.
  - Too **complex** (overfitting) → high variance, poor generalization.
  - **Just right** → neither underfitting nor overfitting, balanced performance.

---

## Examples of Overfitting and Underfitting

### Housing Price Prediction (Linear Regression)
- **Underfitting**: A straight-line fit to the data that doesn’t capture the true relationship.
- **Overfitting**: A high-order polynomial curve that passes through every data point but doesn’t generalize well.

---

## Classification Example (Logistic Regression)
- **Underfitting**: A linear decision boundary that doesn’t separate the data points well.
- **Just Right**: A quadratic decision boundary that fits the data well and generalizes to new examples.
- **Overfitting**: A highly complex, contoured decision boundary that perfectly fits the training data but doesn’t generalize.

---

## Pointers
- **Overfitting** leads to **high variance** (model is too complex).
- **Underfitting** leads to **high bias** (model is too simple).
- The goal in machine learning is to find a model that is neither overfitting nor underfitting, **just right**.
  


---

# Addressing Overfitting



## Causes of Overfitting
- **Overfitting** occurs when a model fits the training data **too well**, including noise and irrelevant patterns.
- It leads to **high variance** and poor **generalization** to new data.

---

## Solutions to Overfitting

### 1. **Collect More Training Data**
- **More data** helps reduce overfitting by providing the model with more examples to learn from.
  - **Effect**: The model fits a smoother function that is less likely to be overly complex.
  - **Limitations**: Sometimes more data is not available (e.g., limited housing market data).

### 2. **Feature Selection**
- **Reducing the number of features** used in the model can help reduce overfitting.
  - **Example**: Instead of using all 100 features, select a subset of the most relevant ones (e.g., size, bedrooms, age).
  - **Challenge**: Selecting too few features might discard useful information.
  - **Automatic Feature Selection**: In future courses, you'll learn algorithms to automatically choose the most relevant features.

### 3. **Regularization**
- **Regularization** reduces the impact of features that have a large effect on the model's parameters without removing them entirely.
  - **How it works**: It **shrinks the parameters** (e.g., w1, w2) to reduce their influence.
  - **Goal**: Prevent overly large parameter values, which often cause overfitting.
  - **Effect**: Even with a complex model (e.g., high-order polynomial), regularization keeps the parameters from becoming excessively large.
  - **Important note**: Regularizing the bias term (b) is usually not necessary.
  






  
---

# Regularization for Overfitting



## Regularization Intuition
- Regularization helps to **shrink model parameters (W1, W2, ..., Wn)** to reduce the complexity of the model and prevent overfitting.
- **Example**: In a high-order polynomial fit, we can make parameters like W3 and W4 **close to zero** to avoid overfitting by penalizing these parameters.

---

## Modified Cost Function
- **Original cost function**: Measures the difference between predictions and actual values (e.g., Mean Squared Error).
- **Modified cost function**: Adds a **regularization term** to the cost function, penalizing large parameter values.
  - **New cost function**:
    ```
    J(θ) = J(original) + λ * (Σ wj²)
    ```
    - `J(original)` is the original cost function (e.g., MSE).
    - `λ` is the regularization parameter.
    - `Σ wj²` is the sum of the squared parameters.

---

## Regularization Mechanism
- **Penalty for large parameters**: By adding the regularization term (e.g., λ * Σ wj²), the model tries to minimize both:
  1. The **original cost** (fit the data well).
  2. The **regularization term** (keep parameters small).
- **λ (Lambda)**: Controls the strength of regularization.
  - If **λ = 0**, no regularization, leading to overfitting.
  - If **λ → ∞**, the model underfits by forcing all parameters close to zero.

---

## Choosing λ (Lambda)
- **Too small λ**: Overfitting (the model fits the data too closely).
- **Too large λ**: Underfitting (the model fails to capture data patterns).
- **Optimal λ**: Balances the fit (minimizing the original cost) and smoothness (minimizing parameter values).

---

## Impact of λ on Model Behavior
1. **λ = 0**: No regularization → **Overfitting** (the model fits overly complex functions).
2. **λ = very large** (e.g., 10¹⁰): **Underfitting** (model parameters are almost zero, resulting in a horizontal line).
3. **Intermediate λ**: Best of both worlds → Model with **smooth complexity**, capturing data patterns without overfitting.




---

# Gradient Descent for Regularized Linear Regression

## Goal
- **Learn how to apply gradient descent** for **regularized linear regression** to reduce overfitting.

---

##  Regularized Linear Regression Cost Function
- **Cost function** now includes an additional **regularization term**:
  ```
  J(θ) = J(original) + λ * (Σ wj²)
  ```
  - **J(original)**: Original cost function (e.g., Mean Squared Error).
  - **λ**: Regularization parameter that controls the penalty on large weights.
  - **wj²**: Square of the model's parameters.

---

## Gradient Descent Update for Regularized Linear Regression
- **Gradient Descent for Regularized Regression**:
  - Update for **w_j** (for j = 1 to n):
    ```
    w_j = w_j - α * (∂J/∂w_j + λ * w_j)
    ```
    - **b** (bias) is updated similarly but without the regularization term.
    - **α**: Learning rate.
    - **λ**: Regularization parameter.

---

## Impact of Regularization on Updates
- Regularization **shrinks** parameters `w_j` on every iteration.
- The term **λ * w_j** added to the gradient descent update encourages smaller values for `w_j`.
  - For each update: 
    ```
    w_j = w_j * (1 - α * λ / m)
    ```
  - This **multiplies** `w_j` by a number slightly **less than 1** on each iteration, gradually shrinking `w_j`.

---

## Intuition Behind Regularization Effect
- **Effect on `w_j`**: By multiplying `w_j` by a small value slightly less than 1, the model **shrinks** `w_j` on each iteration, reducing overfitting.
- For example, with **α = 0.01**, **λ = 1**, and **m = 50**, the term **(1 - α * λ / m)** becomes a value like **0.9998**, reducing `w_j` slightly with each update.

---

## Derivative Calculation (Optional)
- **Derivative of the Cost Function**:
  - The derivative with respect to `w_j` includes the **regularization term**:
    ```
    ∂J/∂w_j = (1/m) * Σ [(w · x + b - y) * x_j] + (λ/m) * w_j
    ```
  - The second term, **(λ/m) * w_j**, comes from the regularization term.

---

# Regularized Logistic Regression - Summary Notes

## Key Concepts
- **Logistic Regression Overfitting**: 
  - Logistic regression can suffer from **overfitting** when using **high-order polynomial features** or many features.
  - Overfitting leads to **complex decision boundaries** that don't generalize well to new data.

- **Regularization in Logistic Regression**: 
  - To reduce overfitting, **regularization** is added to the logistic regression cost function.
  - The modified cost function includes a regularization term:
    ``` 
    J(θ) = J(original) + (λ / 2m) * Σ(wj²)
    ```
  - Where:
    - **λ** is the regularization parameter.
    - **wj²** represents the sum of the squared weights.
    - The goal is to **penalize large weights**, leading to simpler models that generalize better.

---

## Implementation of Regularized Logistic Regression

- **Gradient Descent Update**:
  - The update rule for **w_j** is modified for regularization:
    ```
    w_j = w_j - α * (∂J/∂w_j + λ * w_j)
    ```
  - **For b (bias)**: No change in the update rule for **b** because **b** is not regularized.
  
- **Impact of Regularization**:
  - Regularization **only affects the weights (w_j)**, not the bias (b).
  - The logistic regression model uses the **sigmoid function** rather than a linear function, which makes the update rule similar to that in **regularized linear regression**, but with the sigmoid transformation for **f(z)**.







---

















