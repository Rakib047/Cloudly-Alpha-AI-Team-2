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



