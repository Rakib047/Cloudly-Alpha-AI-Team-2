# Neural Networks


- Neural networks were originally designed to **mimic the human brain**.
- Biological inspiration remains, but modern neural networks have **evolved far** from actual brain processes.

- **History of Neural Networks**:
  - **1950s**: Neural networks research began.
  - **1980s-1990s**: Regained popularity, especially for tasks like **handwritten digit recognition** (e.g., postal codes, checks).
  - **Late 1990s**: Fell out of favor again.
  - **Post-2005**: Resurgence with the term **deep learning**, leading to rapid advancement.

- **Deep Learning Branding**:
  - "Deep learning" sounds more appealing, which helped it **gain attention and momentum**.
  - Deep learning and neural networks often mean **very similar things**.

- **Impact Areas**:
  - Major breakthroughs in:
    - **Speech recognition** (early success)
    - **Computer vision** (e.g., 2012 ImageNet moment)
    - **Natural language processing** (text understanding)
    - **Other fields**: Climate change, medical imaging, advertising, product recommendations, etc.

- **Biological Neurons vs. Artificial Neurons**:
  - **Biological neurons**:
    - Inputs via **dendrites**.
    - Output through an **axon**.
  - **Artificial neurons**:
    - Simplified model.
    - Inputs (numbers) → Computation → Output (number).
    - Connected together to form networks.

- **Importance**:
  - Despite biological inspiration, **we know little** about how the brain really works.
  - Blindly mimicking the brain won't likely lead to real intelligence with current knowledge.
  - Modern deep learning focuses more on **engineering principles** than biological accuracy.

- **Why Neural Networks Took Off Recently**:
  - **Explosion of data**:
    - Society digitized records (health, transactions, etc.), leading to **big data**.
  - **Limitations of traditional algorithms**:
    - Algorithms like **linear regression** and **logistic regression** **couldn’t scale** with more data.
  - **Neural networks scale better**:
    - Larger networks (more neurons) **keep improving** as data increases.
  - **Advances in computing hardware**:
    - Rise of **GPUs** significantly boosted the ability to train large neural networks.
  
- **Performance vs. Data Graph (Conceptual)**:
  - Traditional ML algorithms plateaued with more data.
  - Larger neural networks continued to **improve performance** with more data.

- **Pointers**:
  - Combination of **big data** and **powerful hardware** triggered the **rise of deep learning**.


---

# Demand Prediction Example

## Basic Example: Predicting Top Seller T-Shirts
- **Problem**: Predict if a T-shirt will be a top seller based on its price.
- **Input Feature (x)**: Price of the T-shirt.
- **Output**: Probability of being a top seller (yes/no).
- **Logistic Regression Model**:  
  Formula:  
  ```math
  a = \frac{1}{1 + e^{-(wx + b)}}
  ```
  - ` a ` (activation) represents the output probability.

## Neuron Analogy
- **Neuron as a Computation Unit**: Inputs numbers and outputs numbers.
- **Activation**: Strength of the output signal, inspired by biological neurons.
- **Logistic Regression = Simplified Neuron**: Takes input, computes output using sigmoid activation.

## Building a Neural Network
- **More Complex Example**: 4 input features:
  - Price
  - Shipping cost
  - Marketing expenditure
  - Material quality
- **Important Concepts**:
  - **Affordability**: Based on price and shipping.
  - **Awareness**: Based on marketing.
  - **Perceived Quality**: Based on price and material quality.

- **Three Neurons (Hidden Layer)**:
  - Estimate affordability, awareness, and perceived quality.
- **One Output Neuron (Output Layer)**:
  - Predicts final probability of being a top seller.

## Layer Terminology
- **Input Layer**: Input features (price, shipping, marketing, material).
- **Hidden Layer**: Affordability, awareness, and quality neurons.
- **Output Layer**: Final prediction.

- **Activation Values**:
  - Outputs of neurons in a layer.
- **Hidden Layer**:
  - Called "hidden" because their values are not directly observed in the dataset.

## Full Neural Network Computation
- **Step-by-step**:
  - Input vector ` x ` → Hidden Layer → Output Layer.
  - Each layer transforms its input vector into a new output vector.

## Practical Simplification
- **Full Connectivity**:
  - Each neuron in a hidden layer receives all features from the previous layer.
- **Vector Notation**:
  - Features are packed into a vector ` x ` for simplicity.

## Learning Features Automatically
- **Feature Learning**:
  - Neural networks learn important features (like affordability, awareness) automatically without manual engineering.
- **Comparison**:
  - Like manually engineering new features (e.g., lawn size from house frontage and depth) but automated by the network.

## Neural Network Summary
- **Input Layer**: Vector of features.
- **Hidden Layer**: Computes activation values.
- **Output Layer**: Outputs final prediction.

## Deeper Neural Networks
- **Multiple Hidden Layers**:
  - More complex networks can have several hidden layers.
- **Example Architectures**:
  - Input → Hidden Layer 1 → Hidden Layer 2 → Output
  - Input → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Output
- **Architecture Choices**:
  - Number of layers.
  - Number of neurons per layer.

## Important Terminology
- **Architecture**:
  - Choice of number of layers and neurons per layer.
- **Multilayer Perceptron (MLP)**:
  - Term for a neural network with multiple layers.

---

# Computer Vision Application

## Face Recognition Example
- **Task**: Identify a person from a 1000x1000 pixel image.
- **Image Representation**: 
  - Stored as a 1000x1000 matrix of pixel intensity (brightness) values.
  - Values range from 0 to 255.
  - Example: 197 (top-left pixel), 185 (next pixel), etc.
- **Feature Vector**:
  - Unrolling the matrix creates a vector of 1 million pixel values (1000x1000).

## Neural Network for Face Recognition
- **Input Layer**:
  - Takes the feature vector of 1 million pixel values.
- **Hidden Layers**:
  - Layer 1 extracts basic features.
  - Layer 2 extracts higher-level features.
  - Layer 3 combines features into more abstract representations.
- **Output Layer**:
  - Estimates probability of the person’s identity.

## Visualization of Hidden Layers
- **First Hidden Layer**:
  - Neurons detect simple patterns like short lines or edges.
- **Second Hidden Layer**:
  - Neurons detect face parts like eyes, nose corners, and ears.
- **Third Hidden Layer**:
  - Neurons detect larger structures and complete face shapes.

## Learning Feature Detectors
- **Automatic Learning**:
  - Neural networks learn feature detectors automatically.
  - No manual programming needed to detect edges, face parts, or whole faces.

## Size of Neuron Receptive Fields
- **First Layer Neurons**:
  - Look at small regions of the image.
- **Second Layer Neurons**:
  - Look at larger regions by aggregating smaller features.
- **Third Layer Neurons**:
  - Look at even larger areas to detect complex shapes.

## Generalization to Other Data
- **Training on Different Datasets**:
  - Example: Cars dataset.
  - **First Layer**: Learns edges.
  - **Second Layer**: Learns parts of cars (wheels, windows).
  - **Third Layer**: Learns complete car shapes.

- **Key Point**: 
  - The same network structure learns different feature detectors depending on the dataset.

---

# Building Blocks of Neural Networks - Layers of Neurons


- **Layer of Neurons**:
  - Fundamental building block of modern neural networks.
  - Multiple layers are stacked together to form deep networks.

## Example Setup (Demand Prediction)
- **Input Features**:
  - Four input features provided to a hidden layer of three neurons.
- **Hidden Layer**:
  - Outputs passed to an output layer with one neuron.

## Computation in the Hidden Layer
- **Neurons as Logistic Units**:
  - Each neuron implements a logistic regression function.
- **First Neuron**:
  - Parameters: `w_1`, `b_1`.
  - Output: `a_1 = g(w_1 ⋅ x + b_1)`, where `g` is the sigmoid function.
  - Example output: 0.3.
- **Second Neuron**:
  - Parameters: `w_2`, `b_2`.
  - Output: `a_2 = g(w_2 ⋅ x + b_2)`.
  - Example output: 0.7.
- **Third Neuron**:
  - Parameters: `w_3`, `b_3`.
  - Output: `a_3 = g(w_3 ⋅ x + b_3)`.
  - Example output: 0.2.

## Output of the Hidden Layer
- **Activation Vector**:
  - `[a_1, a_2, a_3]` becomes the input to the next layer (output layer).
- **Layer Notation**:
  - Input layer: **Layer 0**.
  - First hidden layer: **Layer 1**.
  - Output layer: **Layer 2**.
  - Use superscript notation `[1]`, `[2]`, etc., to denote parameters and activations for each layer.

## Computation in the Output Layer
- **Input**:
  - Activation vector $a^[1]$ from the hidden layer.
- **Single Neuron**:
  - Computes $a^[2] = g(w^[2] ⋅ a^[1] + b^[2])$.
  - Example output: 0.84 (probability prediction).
- **Final Output**:
  - $a^[2]$ is a scalar (single number).

## Binary Prediction (Optional Step)
- **Thresholding**:
  - If $a^[2] > 0.5$, predict `ŷ = 1`.
  - Else, predict `ŷ = 0`.
- **Usage**:
  - Converts probability into a binary decision (e.g., top seller or not).

## Pointers
- **Layer Computation**:
  - Each layer processes an input vector through logistic units.
  - Produces an output vector passed to the next layer.
- **Deep Networks**:
  - Larger neural networks have many such layers.
---

# Building a Deeper Neural Network - Layer Computations


- **From Single Layer to Deep Networks**:
  - A layer inputs a vector of numbers and outputs another vector.
  - In deeper networks, this idea is stacked multiple times.

## Example Network Structure
- **Layers**:
  - Input Layer (Layer 0): Not counted in the number of layers.
  - Layers 1, 2, 3: Hidden Layers.
  - Layer 4: Output Layer.
- **Counting Layers**:
  - Neural network is said to have 4 layers (hidden + output).

## Zooming into Layer 3 (Final Hidden Layer)
- **Inputs**:
  - Takes in activation vector `a^[2]` (output of Layer 2).
- **Outputs**:
  - Produces activation vector `a^[3]`.
- **Neuron Computation in Layer 3**:
  - For each neuron:
    - `a_j^[3] = g(w_j^[3] ⋅ a^[2] + b_j^[3])`
    - `g` is the sigmoid activation function.

## Notation Recap
- **Superscripts and Subscripts**:
  - Superscript `[l]` denotes the layer number.
  - Subscript `j` denotes the neuron/unit number in that layer.
- **Correct Computation for Neuron 2 in Layer 3**:
  - ```math
    a_2^[3] = g(w_2^[3] ⋅ a^[2] + b_2^[3])
    ```
- **Important Reminder**:
  - Inputs to a layer come from the full activation vector of the previous layer, not a single value.

## General Equation for Any Layer
- **For Any Layer `l` and Unit `j`**:
  - ```math
    a_j^[l] = g(w_j^[l] ⋅ a^[l-1] + b_j^[l])
    ```
- **Key Points**:
  - $w_j^[l]$: Weights for neuron `j` in layer `l`.
  - $a^[l-1]$: Activation vector from previous layer (`l-1`).
  - `g`: Activation function (e.g., sigmoid).

## Units vs Neurons
- **Terminology**:
  - Units and neurons are used interchangeably.
  - Each unit corresponds to a single neuron in a layer.

## Activation Function
- **Definition**:
  - Function `g` that produces the activations.
- **Sigmoid Function**:
  - So far, only sigmoid is used.
  - Other activation functions will be introduced later.

## Final Notation Consistency
- **Input Vector `X`**:
  - Also called `a^[0]`.
- **Implication**:
  - Allows the general activation formula to apply even for the first hidden layer.

## Pointers
- **Layer Computations**:
  - Given parameters and previous layer activations, can compute any layer’s activations.
---

# Forward Propagation in Neural Networks



## Example Setup
- **Input Image**: 8x8 grayscale image → 64 pixel intensity values (0 = black, 255 = white).
- **Input Vector `x`**: 64 features.
- **Neural Network Architecture**:
  - **Hidden Layer 1**: 25 units (neurons).
  - **Hidden Layer 2**: 15 units.
  - **Output Layer**: 1 unit → outputs probability of being digit **1**.

---

## Computation Steps

### Step 1: Compute Activations of Layer 1
- Formula:
```math  
  a^[1] = g(W^[1] ⋅ a^[0] + b^[1])
  ```
- Notes:
  - `a^[0] = x` (input vector).
  - Output `a^[1]` has 25 values (one per unit).
  - Parameters: `W^[1]`, `b^[1]` for 25 units.

### Step 2: Compute Activations of Layer 2
- Formula:
```math  
  a^[2] = g(W^[2] ⋅ a^[1] + b^[2])
```
- Notes:
  - Output `a^[2]` has 15 values (one per unit).
  - Parameters: `W^[2]`, `b^[2]` for 15 units.

### Step 3: Compute Activation of Output Layer
- Formula:
```math  
  a^[3] = g(W^[3] ⋅ a^[2] + b^[3])
```
- Notes:
  - Single output: scalar value between 0 and 1.
  - Represents predicted probability of the digit being **1**.

---

## Final Output
- **Prediction**:
  - `a^[3]` (or `f(x)`) is the neural network's prediction.
  - Thresholding:
    - If `a^[3] > 0.5`, predict **1**.
    - Else, predict **0**.

---

## Terminology and Concepts

- **Forward Propagation**:
  - The process of computing activations from input layer to output layer.
  - Computation flows **forward** (left to right).
- **Function Notation**:
  - `f(x)` is used to denote the final output of the neural network.
- **Architecture Note**:
  - Common design: more neurons in early layers, fewer in later layers.

---

## Practical Usage
- With forward propagation, you can:
  - Use pretrained neural networks downloaded from the internet.
  - Perform inference on new input data.

---

# Implementing Neural Network Inference with TensorFlow


- **Framework Focus**: TensorFlow (most used in this specialization).
- **Alternative**: PyTorch (popular but not covered in detail here).
- **Topic**: Implementing inference (forward propagation) in TensorFlow.

---

## Example 1: Coffee Bean Roasting

### Task
- Predict if coffee is **good or bad** based on:
  - **Temperature** (°C)
  - **Duration** (minutes)

### Dataset Insight
- **Input Features**: 2D feature vector `[temperature, duration]`
- **Labels**:
  - `y = 1`: Good coffee
  - `y = 0`: Bad coffee
- **Pattern**:
  - Undercooked if temp/duration is too low
  - Overcooked if temp/duration is too high
  - Only optimal range (triangle-shaped area) produces good coffee

### Inference in TensorFlow
1. **Input Vector**:  
   ```python
   x = np.array([200, 17])

2. **First Hidden Layer**:  
   ```python
   layer1 = tf.keras.layers.Dense(units=3, activation='sigmoid')
   a1 = layer1(x)

3. **Second Hidden Layer (Output Layer)**:  
   ```python
    layer2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
    a2 = layer2(a1)

4. **Thresholding**
    ```python
    y_hat = 1 if a2 >= 0.5 else 
    
## Handwritten Digit Classification with TensorFlow

### Task
- Classify an image of a handwritten digit as either `0` or `1` (binary classification).

### Input
- `x = np.array([...])`
  - A list of pixel intensity values (e.g., 64 values for an 8x8 image).
  - This serves as the input feature vector to the neural network.

### Neural Network Architecture
- 3 layers:
  - **Layer 1**: 25 units, sigmoid activation
  - **Layer 2**: 15 units, sigmoid activation
  - **Layer 3**: 1 unit, sigmoid activation (final output)

### Forward Propagation Steps

    ```python
    import numpy as np
    import tensorflow as tf

    # Input: example image represented as a NumPy array
    x = np.array([...])  # Replace with actual pixel values

    # Layer 1: First hidden layer with 25 units
    layer1 = tf.keras.layers.Dense(units=25, activation='sigmoid')
    a1 = layer1(x)

    # Layer 2: Second hidden layer with 15 units
    layer2 = tf.keras.layers.Dense(units=15, activation='sigmoid')
    a2 = layer2(a1)

    # Layer 3: Output layer with 1 unit
    layer3 = tf.keras.layers.Dense(units=1, activation='sigmoid')
    a3 = layer3(a2)

    # Final prediction: binary classification
    y_hat = 1 if a3 >= 0.5 else 0
    ```

---

# Data Representation in NumPy vs TensorFlow



---

## Matrix Representation in NumPy

- **Matrix Dimensions**: Described as `rows x columns`.
  - Example: `2 x 3` matrix = 2 rows, 3 columns.
    ```python
    x = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
    ```

- **4 x 2 Matrix**:
    ```python
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Shape: (4, 2)
    ```

- **1 x 2 Row Vector**:
    ```python
    x = np.array([[200, 17]])  # Shape: (1, 2)
    ```

- **2 x 1 Column Vector**:
    ```python
    x = np.array([[200], [17]])  # Shape: (2, 1)
    ```

- **1D Array (No explicit rows/columns)**:
    ```python
    x = np.array([200, 17])  # Shape: (2,)
    ```

---

## TensorFlow's Representation

- TensorFlow prefers **explicit 2D arrays**, even for single examples.
- Designed this way for **efficiency on large datasets**.

- **double square brackets:**
  - TensorFlow expects data to be shaped as **matrices**: one row per example, one column per feature.
    ```python
    x = np.array([[200, 17]])  # Shape: (1, 2)
    ```

- In TensorFlow, a **tensor** is just a generalization of a matrix (n-dimensional array), used for efficient computation.

---

## Neural Network Forward Propagation (TensorFlow Example)

1. **First Layer Activation (`a1`)**:
    ```python
    a1 = layer1(x)
    print(a1)
    # Output: tf.Tensor([[0.2, 0.7, 0.3]], shape=(1, 3), dtype=float32)
    ```
   - Shape: `(1, 3)` → One example, 3 hidden units.
   - `float32`: 32-bit floating-point numbers.

2. **Second Layer Activation (`a2`)**:
    ```python
    a2 = layer2(a1)
    print(a2)
    # Output: tf.Tensor([[0.8]], shape=(1, 1), dtype=float32)
    ```
   - Shape: `(1, 1)` → Single output from sigmoid unit.

---

## TensorFlow Tensor ↔ NumPy Array

- **Convert Tensor to NumPy**:
    ```python
    a1.numpy()  # Returns NumPy array
    a2.numpy()
    ```

- Internally, TensorFlow operates on tensors but allows you to pass in NumPy arrays — converts them automatically.

---

## Key Points

- **NumPy**:
  - Uses both 1D and 2D arrays.
  - A 1D vector like `np.array([200, 17])` has no explicit rows/columns.

- **TensorFlow**:
  - Uses 2D matrices (tensors) by default, even for single samples.
  - More efficient for internal operations.

- **Shape Importance**:
  - Always use **explicit 2D shape** for inputs in TensorFlow: `(num_examples, num_features)`.
  - A row vector is shape `(1, n)` and a column vector is `(n, 1)`, but a 1D array is just `(n,)`.

- **Tensor ≈ Matrix**:
  - Think of a TensorFlow tensor as just a matrix (for this course).
  - Can be converted back and forth using `.numpy()`.

---

# TensorFlow Neural Network Construction



## Forward Propagation: Manual Approach
- **Manual Layer-by-Layer Setup**:  
  Previously, forward propagation was done step-by-step:
  - Define `x`
  - Create `layer1`, compute `a1`
  - Create `layer2`, compute `a2`
- **Explicit Computation**:  
  Each layer's computation was done manually by chaining `.apply()` methods or similar.

---

## Forward Propagation: Sequential Model
- **Sequential API**:  
  TensorFlow's `Sequential` API allows chaining layers together to form a model:
  ```python
  model = tf.keras.Sequential([layer1, layer2])
  ```
  This handles forward propagation internally.

- **Training Data Format**:
  - **Input Features (`X`)**: Stored in a NumPy 2D array (e.g., shape `(4, 2)` for 4 examples with 2 features).
  - **Target Labels (`Y`)**: Stored as a 1D NumPy array (e.g., `[1, 0, 0, 1]`).

- **Training the Model**:
  - **Compile the Model**:
    ```python
    model.compile(...)  # Details covered in future lessons
    ```
  - **Fit/Train the Model**:
    ```python
    model.fit(X, Y)
    ```

- **Inference / Prediction**:
  - For new input `X_new`, simply call:
    ```python
    model.predict(X_new)
    ```

---

## Streamlined Layer Definition (Best Practice)
- **Avoid Explicit Layer Variables**:  
  Instead of assigning each layer to a separate variable, pass them directly into `Sequential`:
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  ```

- **Compact and Clean Code**:  
  This is a widely accepted TensorFlow coding convention and is functionally equivalent.

---

## Application to Digit Classification Example
- **Similar Refactoring**:
  - Define `layer1`, `layer2`, `layer3`
  - Use them in `Sequential`
  - Compile and train using `.compile()` and `.fit()`
  - Predict using `.predict()`

- **Final Version**:
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(...),  # layer1
      tf.keras.layers.Dense(...),  # layer2
      tf.keras.layers.Dense(...)   # layer3
  ])
  ```



---

# Forward Propagation from Scratch in Python



## Motivation
- **Why Implement from Scratch?**  
  To gain deep understanding of:
  - How data flows through a neural network.
  - The mathematical operations underlying each layer.
  - How activations are computed.

- **Practical Use**:  
  Although rarely needed in real-world practice, knowing this builds strong intuition and prepares you for:
  - Debugging frameworks.
  - Designing custom layers or models.
  - Possibly building better tools in the future.

---

## Notation and Setup
- **Coffee Roasting Example**:  
  A toy dataset with 2 input features used to demonstrate the flow.

- **1D Array Convention**:  
  Vectors (inputs, weights, biases) are represented using **1D NumPy arrays** (e.g., `[w1, w2]`) for simplicity.

---

## Step-by-Step Forward Propagation

### Layer 1 (Hidden Layer with 3 Units)

- **Compute Activation a1_1**:
  - Inputs:  
    `w1_1 = np.array([1, 2])`, `b1_1 = -1`, `x = np.array([x1, x2])`
  - Compute:  
    `z1_1 = np.dot(w1_1, x) + b1_1`  
    `a1_1 = sigmoid(z1_1)`

- **Compute Activation a1_2**:
  - Inputs:  
    `w1_2 = np.array([-3, 4])`, `b1_2 = 1`
  - Compute:  
    `z1_2 = np.dot(w1_2, x) + b1_2`  
    `a1_2 = sigmoid(z1_2)`

- **Compute Activation a1_3**:
  - Inputs:  
    `w1_3 = np.array([1, -1])`, `b1_3 = 0`
  - Compute:  
    `z1_3 = np.dot(w1_3, x) + b1_3`  
    `a1_3 = sigmoid(z1_3)`

- **Group into Activation Vector a1**:
  ```python
  a1 = np.array([a1_1, a1_2, a1_3])
  ```

---

### Layer 2 (Output Layer with 1 Unit)

- **Compute Final Activation a2**:
  - Inputs:  
    `w2_1 = np.array([2, -1, 1])`, `b2_1 = -3`
  - Compute:  
    `z2_1 = np.dot(w2_1, a1) + b2_1`  
    `a2 = sigmoid(z2_1)`

---

## Notes on Implementation
- **Sigmoid Function**:
  ```python
  def sigmoid(z):
      return 1 / (1 + np.exp(-z))
  ```

- **Final Output**:
  - `a2` is the prediction/output of the network for input `x`.
  - This mirrors what `model.predict()` does under-the-hood in TensorFlow.

---


- **Manual forward propagation** is about computing each neuron's output via dot products and applying activation functions.
- Though verbose, it builds deep understanding of:
  - Data flow
  - Matrix/vector operations
  - The role of weights and biases



```python

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Input
x = np.array([x1, x2])

# Layer 1
w1_1, b1_1 = np.array([1, 2]), -1
a1_1 = sigmoid(np.dot(w1_1, x) + b1_1)

w1_2, b1_2 = np.array([-3, 4]), 1
a1_2 = sigmoid(np.dot(w1_2, x) + b1_2)

w1_3, b1_3 = np.array([1, -1]), 0
a1_3 = sigmoid(np.dot(w1_3, x) + b1_3)

a1 = np.array([a1_1, a1_2, a1_3])

# Layer 2
w2_1, b2_1 = np.array([2, -1, 1]), -3
a2 = sigmoid(np.dot(w2_1, a1) + b2_1)

```
---

# General Forward Propagation in Python using a Dense Layer Function



## Key Concepts

- **Dense Layer Function**:
  A modular function that computes the output (activations) of a single dense (fully connected) layer given:
  - Input activations.
  - Weight matrix `W`.
  - Bias vector `b`.

- **Weight Matrix (W)**:
  Each **column** corresponds to weights of one neuron in the current layer.  
  E.g., for 3 neurons with 2 input features each:  
  ```text
  W = [
    [w1_1, w1_2, w1_3],
    [w2_1, w2_2, w2_3]
  ]
  ```

- **Bias Vector (b)**:
  Each entry corresponds to the bias for one neuron.  
  E.g., `b = np.array([-1, 1, 2])`

- **Activation Vector (a)**:
  Output of the dense layer, computed by applying the sigmoid function to each neuron's weighted input (`z`).

---

## Dense Layer Code Breakdown

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense(a_prev, W, b):
    units = W.shape[1]             # Number of neurons in current layer
    a = np.zeros(units)            # Initialize output activations
    
    for j in range(units):
        w = W[:, j]                # Get j-th column (weights for j-th neuron)
        z = np.dot(w, a_prev) + b[j]  # Compute z = w^T * a + b
        a[j] = sigmoid(z)          # Apply activation
    return a
```

---

## How to Use the Dense Function for Forward Propagation

Assuming a 4-layer network (input + 3 layers):

```python
# Input
x = np.array([...])  # Input feature vector

# Layer 1 parameters
W1 = np.array([[...], [...]]).T  # Shape: (input_size, units)
b1 = np.array([...])

# Layer 2 parameters
W2 = np.array([[...], [...], [...]]).T
b2 = np.array([...])

# Layer 3 parameters
W3 = ...
b3 = ...

# Forward propagation
a1 = dense(x, W1, b1)
a2 = dense(a1, W2, b2)
a3 = dense(a2, W3, b3)

# Final output
f_x = a3
```

---



- **Dense function** abstracts away individual neuron logic.
- Enables construction of deep networks with multiple layers by calling `dense()` repeatedly.
- Important for debugging real models in TensorFlow/PyTorch.
- Understanding how activations are computed helps:
  - Identify bottlenecks
  - Fix vanishing gradients
  - Tune network depth and width

---



Even when using high-level frameworks:
- Knowing the mechanics of forward propagation lets you debug issues like:
  - Incorrect dimensions
  - Dead neurons (all outputs 0 or 1)
  - Slow training
- Empowers you to design and test novel network architectures.

---

# AGI and the Future of Artificial Intelligence

## Dream of AGI
- **Inspiration from Youth**: The idea of creating an AI as intelligent as a human has been a long-standing dream for many researchers, including the speaker.
- **Unclear Timeline**: Path to AGI (Artificial General Intelligence) is uncertain — may take decades, centuries, or longer.

## ANI vs AGI
- **ANI (Artificial Narrow Intelligence)**:
  - Performs specific tasks extremely well (e.g., smart speakers, self-driving cars, web search).
  - Subset of AI and has made tremendous progress recently.
- **AGI (Artificial General Intelligence)**:
  - Aspires to replicate human-like intelligence capable of generalizing across many tasks.
  - Progress towards AGI is unclear despite ANI advancements.
- **Hype Explained**: Progress in ANI has led to confusion — some believe rapid ANI progress implies similar AGI progress, which is likely incorrect.

## Misconceptions About Simulating the Brain
- **Rise of Deep Learning**: Led to hopes that simulating many neurons could recreate human brain functionality.
- **Two Key Problems**:
  1. Artificial neurons (e.g., logistic units) are far simpler than biological neurons.
  2. Limited understanding of how the brain actually works — neuron input-output mapping is still largely unknown.

## Brain Plasticity Experiments
- **One Learning Algorithm Hypothesis**:
  - Suggests that one or a few algorithms might underlie much of human intelligence.
  - Inspired by how different brain regions can adapt to new data types.

### Supporting Experiments:
- **Auditory Cortex Rewiring**: When fed visual inputs, the auditory cortex learns to see.
- **Somatosensory Cortex Rewiring**: Touch-processing brain areas can adapt to visual inputs.
- **Cross-modal Adaptation**: Different parts of the brain can adapt based on the input they receive.

### Technological Examples:
- **Camera to Tongue Interface**: Devices that convert visual data into tongue vibrations help blind individuals "see".
- **Human Echolocation**: Humans can be trained to use sonar-like clicking sounds to navigate.
- **Haptic Belt**: Wearable device vibrates in the northward direction, giving users a sense of direction (like a natural compass).
- **Third Eye Implant**: Frog brains adapted to use an additional implanted eye, showing high brain adaptability.

## Key Takeaways from Neuroscience
- **Brain Plasticity**: The brain is highly adaptable (also called “plastic”) and can process new sensory data types.
- **Open Question**: If the same brain tissue can adapt to different modalities, is there a universal learning algorithm?
- **Challenge**: Even if there is such an algorithm, we currently don’t know what it is.

## Personal Reflections and Future Direction
- **Hope for AGI**: Despite the challenges and uncertainties, the hope for discovering a path to AGI remains alive.
- **Avoiding Hype**: It's important to stay realistic — we don’t yet know the path or the algorithm to AGI.
- **Neural Networks Today**: Regardless of AGI, neural networks are still extremely powerful tools for solving practical problems.



---

# Vectorized Implementation of Neural Networks

## Importance of Vectorization
- **Key to Scaling Deep Learning**: Vectorization enables efficient computation, allowing neural networks to scale to very large models.
- **Hardware Efficiency**: Matrix multiplications can be executed very efficiently on parallel computing hardware like GPUs and some CPUs.
- **Success of Deep Learning**: Without vectorized computation, deep learning would not have achieved its current scale or success.

## Forward Propagation: Non-Vectorized vs. Vectorized
- **Traditional (Non-Vectorized) Approach**:
  - Involves explicit for-loops to compute outputs from inputs, weights, and biases.
  - Slower and less efficient due to iterative computation.

- **Vectorized Implementation**:
  - Uses matrix operations to replace loops and batch process inputs.
  - More concise, faster, and better optimized for parallel hardware.

## Example Code Explanation
- **Inputs and Parameters**:
  - `X`: Input vector, reshaped as a 2D array (e.g., `[[x1, x2]]`)
  - `W`: Weight matrix (dimensions suitable for matmul with `X`)
  - `B`: Bias vector (also a 2D array, e.g., `[[b1, b2, b3]]`)

- **Forward Propagation Steps**:
  1. **Matrix Multiplication**:  
     `Z = np.matmul(X, W) + B`  
     - Computes the linear combination of inputs and weights.
  2. **Activation Function**:  
     `A_out = g(Z)`  
     - Applies element-wise activation (e.g., sigmoid) to `Z`.

- **Efficiency**:
  - All intermediate values (`X`, `W`, `B`, `Z`, `A_out`) are 2D matrices.
  - Significantly faster than using explicit loops.

---

# Matrix Multiplication and Dot Products

## Understanding Dot Products
- **Dot Product of Two Vectors**:
  - Example: `a = [1, 2]`, `w = [3, 4]`
  - Dot product: `z = 1*3 + 2*4 = 3 + 8 = 11`
  - Formula: `z = a₁ * w₁ + a₂ * w₂ + ... + aₙ * wₙ`

- **Alternate Representation with Transpose**:
  - A vector `a` can be written as a column vector.
  - Its transpose, `aᵀ`, becomes a row vector.
  - Dot product can be computed as matrix multiplication:  
    `z = aᵀ * w`

## Vector-Matrix Multiplication
- **Setup**:
  - Vector `a = [1, 2]` → `aᵀ = [1, 2]` (1×2)
  - Matrix `W = [[3, 5], [4, 6]]` (2×2)

- **Computation**:
  - `Z = aᵀ * W` → (1×2 matrix result)
  - First element: `1*3 + 2*4 = 11`
  - Second element: `1*5 + 2*6 = 17`
  - Result: `Z = [[11, 17]]`

## Generalizing to Matrix-Matrix Multiplication
- **Matrix A**:
  ```
  A = [[1, -1],
       [2, -2]]
  ```
  - Columns: `a₁ = [1, 2]`, `a₂ = [-1, -2]`
  - Transpose `Aᵀ = [[1, 2], [-1, -2]]`

- **Matrix W**:
  ```
  W = [[3, 5],
       [4, 6]]
  ```
  - Columns: `w₁ = [3, 4]`, `w₂ = [5, 6]`

- **Compute `Aᵀ * W`**:
  - Multiply each row of `Aᵀ` (i.e., each transposed column of `A`) with `W`
  - Row 1 (`a₁ᵀ`):
    - Dot with `w₁`: `1*3 + 2*4 = 11`
    - Dot with `w₂`: `1*5 + 2*6 = 17`
    - → First row of result: `[11, 17]`
  - Row 2 (`a₂ᵀ`):
    - Dot with `w₁`: `-1*3 + -2*4 = -3 - 8 = -11`
    - Dot with `w₂`: `-1*5 + -2*6 = -5 -12 = -17`
    - → Second row of result: `[-11, -17]`

- **Final Result**:
  ```
  Aᵀ * W = [[11, 17],
            [-11, -17]]
  ```

## Conceptual Understanding
- **Matrix as Collection of Vectors**:
  - View matrix columns as individual vectors.
  - Transposed matrix rows correspond to those column vectors laid sideways.

- **Matrix Multiplication = Many Dot Products**:
  - Each element in the result is a dot product of a row of the first matrix and a column of the second.
  - Multiply rows from `Aᵀ` with columns from `W`.


---

# Matrix Multiplication (General Form)

## Matrix Setup

- **Matrix A (2×3)**:
  - Two rows, three columns.
  - Columns treated as vectors: **a₁, a₂, a₃**.

- **Matrix Aᵗ (Transpose of A)**:
  - Becomes a 3×2 matrix.
  - Columns of A become rows of Aᵗ: **a₁ᵗ, a₂ᵗ, a₃ᵗ**.

- **Matrix W (2×4)**:
  - Two rows, four columns.
  - Columns treated as vectors: **w₁, w₂, w₃, w₄**.

---

## Matrix Multiplication: Aᵗ × W

- Resulting matrix **Z** has shape **3×4**.
- Each element of Z is computed using a **dot product** of a row of Aᵗ and a column of W.

---

## Example Computations

- **Z[0][0]** (1st row, 1st col):
  - Dot product of Aᵗ row 1 (a₁ᵗ = [1, 2]) and W col 1 (w₁ = [3, 4]):
  - (1×3) + (2×4) = 3 + 8 = **11**

- **Z[2][1]** (3rd row, 2nd col):
  - Dot product of Aᵗ row 3 (a₃ᵗ = [0.1, 0.2]) and W col 2 (w₂ = [5, 6]):
  - (0.1×5) + (0.2×6) = 0.5 + 1.2 = **1.7**

- **Z[1][2]** (2nd row, 3rd col):
  - Dot product of Aᵗ row 2 (a₂ᵗ = [-1, -2]) and W col 3 (w₃ = [7, 8]):
  - (-1×7) + (-2×8) = -7 - 16 = **-23**

---

## Dimensional Rules for Matrix Multiplication

- **Requirement**:
  - Inner dimensions must match:  
    If Aᵗ is **m×n** and W is **n×p**, then **n = n** must hold.

- **Why**:
  - You can only take dot products between vectors of **equal length**.

- **Output Shape**:
  - If Aᵗ is **m×n** and W is **n×p**, the result Z is **m×p**:
    - **Rows of Aᵗ**
    - **Columns of W**

---

# Vectorized Implementation of a Neural Network

## Matrix Multiplication Recap

- You previously computed `Z = Aᵗ × W` to understand matrix multiplication.
- In NumPy:
  - Transpose matrix: `AT = A.T`
  - Multiply matrices: `Z = np.matmul(AT, W)`
  - Alternative syntax: `Z = AT @ W` (less preferred for clarity)

---

## Forward Propagation: Vectorized Code

### Inputs and Parameters
- **A_in (input features)**: 1×2 matrix (e.g., roasting temp 200, time 17)
  ```python
  A_in = np.array([[200, 17]])
  ```
- **W (weights)**: 2×3 matrix (each column is a weight vector for a neuron)
  ```python
  W = np.array([[w1_1, w2_1, w3_1],
                [w1_2, w2_2, w3_2]])
  ```
- **B (biases)**: 1×3 matrix
  ```python
  B = np.array([[b1, b2, b3]])
  ```

### Computation of Z
- Formula:  
  ```python
  Z = np.matmul(A_in, W) + B
  ```
- Each value in Z corresponds to:  
  `zⱼ = (A_in ⋅ wⱼ) + bⱼ`
- Example results:
  - `z₁ = 165`
  - `z₂ = -531`
  - `z₃ = 900`

---

## Activation Step

- Apply **sigmoid function `g(z)`** element-wise:
  ```python
  A_out = sigmoid(Z)
  ```
- Result:
  - `sigmoid(165) ≈ 1`
  - `sigmoid(-531) ≈ 0`
  - `sigmoid(900) ≈ 1`
- Final output: `A_out = [1, 0, 1]`

---



```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

A_in = np.array([[200, 17]])     # input features
W = np.array([[w1, w2, w3],      # weight matrix (2x3)
              [w4, w5, w6]])
B = np.array([[b1, b2, b3]])     # bias matrix (1x3)

Z = np.matmul(A_in, W) + B       # linear computation
A_out = sigmoid(Z)               # activation function
```

---

# Week 2: Training a Neural Network

## From Inference to Training

- **Last week**: Focused on inference – computing outputs given inputs and model parameters.
- **This week**: Learn how to **train** a neural network using your own data.

---

## Example: Handwritten Digit Classification

- Goal: Classify an image as a `0` or `1`.
- Architecture:
  - Input layer: image `X`
  - Hidden layer 1: 25 units (sigmoid)
  - Hidden layer 2: 15 units (sigmoid)
  - Output layer: 1 unit (sigmoid for binary classification)
- Given a dataset `X` (images) and `Y` (labels), how to learn the parameters?

---

## TensorFlow Code to Train the Network

### Define the Model Architecture

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(25, activation='sigmoid'),  # Hidden Layer 1
    keras.layers.Dense(15, activation='sigmoid'),  # Hidden Layer 2
    keras.layers.Dense(1, activation='sigmoid')    # Output Layer
])
```

### Compile the Model

- Define the loss function and optimizer
- Use **binary crossentropy** for binary classification

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

### Fit the Model to the Training Data

- `model.fit(X, Y, epochs=n)` runs the training loop
- `epochs`: how many full passes (iterations) over the training data

```python
model.fit(X, Y, epochs=10)
```

---


1. **Model Definition**: Describe how the model processes inputs.
2. **Model Compilation**: Specify the loss function (e.g., binary crossentropy).
3. **Model Training**: Use `.fit()` to apply the training process using the dataset.

---

# Training a Neural Network 



## TensorFlow Training Process (3 Steps)

### Step 1: Specify the Model
- Use `tf.keras.Sequential()` to stack layers:
  - First hidden layer: 25 units, sigmoid activation.
  - Second hidden layer: 15 units, sigmoid activation.
  - Output layer: 1 unit, sigmoid activation.
- Defines the forward propagation / inference path.
- Tells TensorFlow how to compute output `f(x)` from input `x` and parameters `W`, `b`.

### Step 2: Compile the Model
- Use `model.compile()` to:
  - Set the **loss function**: `binary_crossentropy` for binary classification.
    - Formula: `−y log(f(x)) − (1 − y) log(1 − f(x))`
    - Same loss function used in logistic regression.
  - Optionally specify the optimizer and evaluation metrics (e.g., accuracy).
- The **cost function** is defined as the average loss across the training examples.

### Step 3: Fit the Model
- Use `model.fit(X, Y, epochs=100)` to train.
  - `epochs` define how many iterations of gradient descent to perform.
- TensorFlow handles gradient descent internally.
  - Uses **backpropagation** to compute gradients.
  - Uses optimized versions of gradient descent (e.g., Adam, SGD).

---

## Conceptual Mapping to Logistic Regression

1. **Forward Pass**:
   - Logistic Regression: `f(x) = sigmoid(wᵗx + b)`
   - Neural Network: multiple layers of `z = wᵗx + b` passed through activations (e.g., sigmoid)

2. **Loss Function**:
   - Same binary cross-entropy loss for logistic regression and neural network binary classification.

3. **Optimization**:
   - Logistic regression and neural networks both minimize cost using gradient descent.
   - Neural networks use backpropagation to calculate gradients for all parameters.

---

## Loss Functions
- **Binary Crossentropy**:
  - Used for binary classification.
  - TensorFlow name: `BinaryCrossentropy`

- **Mean Squared Error (MSE)**:
  - Used for regression tasks.
  - TensorFlow name: `MeanSquaredError`
  - Formula: `(1/2)(f(x) - y)²`

---

## Parameters Notation
- **W, b**: Parameters of the model.
  - In neural networks: `W1, W2, ..., WL` and `b1, b2, ..., bL`
- **f(x)**: Output of the neural network.
  - Can also be written as `f(W, b; x)` to show dependency on parameters.



---


- Similar to how libraries now handle:
  - Sorting
  - Square roots
  - Matrix multiplication
- Deep learning libraries (like TensorFlow/PyTorch) are now standard for implementing neural networks.
- Most commercial implementations rely on these libraries rather than hand-coded solutions.

---

# Activation Functions in Neural Networks

## Different Activation Functions
- **Initial Approach**: Used sigmoid activation in all nodes, inspired by logistic regression.
- **Limitation**: Sigmoid maps values between 0 and 1, which may not model real-world scenarios with a wider range effectively.

## Real-World Scenario Example
- **Demand Prediction Example**:
  - Predict if a product is a top seller based on features like price, marketing, shipping, perceived quality.
  - Awareness was previously modeled as binary (aware or not aware).
  - **Issue**: Awareness is more realistically a spectrum (e.g., low awareness to viral popularity).
  - **Solution**: Need activation functions that allow non-binary, unbounded outputs.

## ReLU (Rectified Linear Unit)
- **Function Definition**: 
  ```math
  g(z) = max(0, z)
  ```
- **Graph**:
  - For z < 0: output is 0.
  - For z ≥ 0: output is z (a 45° line).
- **Behavior**:
  - Allows activations to be 0 or any positive number.
  - Helps model non-binary features like degrees of awareness.
- **Name**: ReLU = Rectified Linear Unit
  - Name origin is not important; just remember it's widely used.

## Common Activation Functions
- **Sigmoid Function**:
  - Squashes input into a range between 0 and 1.
  - Useful in output layers for binary classification.

- **ReLU Function**:
  - Outputs 0 for negative inputs, and linear for positive inputs.
  - Encourages sparse activations and reduces vanishing gradients.

- **Linear Activation Function**:
  - Defined as:
    ```math
    g(z) = z
    ```
  - Often used in output layers for regression problems.
  - Equivalent to using **no activation function**.

## Terminology
- Some refer to linear activation as "no activation."
- This class uses the term **linear activation function** instead of "no activation."

---

# Choosing Activation Functions in Neural Networks

## Output Layer Activation Function

- **Binary Classification (y ∈ {0,1})**
  - **Use**: Sigmoid Activation Function
  - **Why**: Outputs a probability between 0 and 1; ideal for predicting class probabilities like logistic regression.

- **Regression (y ∈ ℝ, can be positive or negative)**
  - **Use**: Linear Activation Function (`g(z) = z`)
  - **Why**: Allows predictions over the full range of real numbers.

- **Regression (y ≥ 0, only non-negative values)**
  - **Use**: ReLU Activation Function (`g(z) = max(0, z)`)
  - **Why**: Output restricted to zero or positive values; ideal when predicting quantities that can't be negative (e.g., house prices).

> General Tip: Choose output layer activation based on the type of the target variable `y`.

## Hidden Layer Activation Function

- **Default Recommendation**: Use ReLU
  - **Why**:
    - **Efficiency**: Faster to compute than sigmoid (no exponentials).
    - **Better Gradient Flow**: Sigmoid goes flat in both tails (vanishing gradient issue), while ReLU only flattens on the left side.
    - **Faster Learning**: Fewer flat regions in cost function J(W, b), leading to faster convergence in training.

- **Historical Note**:
  - Earlier networks often used sigmoid activations.
  - Modern practice heavily favors ReLU due to practical advantages.

- **TensorFlow Syntax Example**:
  ```python
  # First hidden layer with ReLU
  Dense(units=..., activation='relu')

  # Output layer examples:
  Dense(units=1, activation='sigmoid')  # for binary classification
  Dense(units=1, activation='linear')   # for regression (y ∈ ℝ)
  Dense(units=1, activation='relu')     # for non-negative regression (y ≥ 0)
  ```
---

## Additional Activation Functions (Optional / Advanced)

- **Tanh**: Squashes input between -1 and 1; historically popular but now less common.
- **LeakyReLU**: Like ReLU but allows small negative outputs to address the "dying ReLU" problem.
- **Swish**: A newer function by Google researchers that can sometimes outperform ReLU.

> While newer functions like LeakyReLU or Swish exist, ReLU remains sufficient for most applications.

## Activation Function Choices

| Problem Type                  | Recommended Activation Function |
|------------------------------|---------------------------------|
| Binary Classification        | Sigmoid                         |
| Regression (any real value)  | Linear                          |
| Regression (non-negative)    | ReLU                            |
| Hidden Layers (default)      | ReLU                            |

>  Tip: When in doubt, use **ReLU** for hidden layers and choose the output activation based on the label `y`.

---

# Why Neural Networks Need Activation Functions

## Problem with Using Only Linear Activation

- **Linear Activation Function (g(z) = z) in Every Neuron**
  - Makes the entire network behave like a **linear model**.
  - Loses the ability to model complex, non-linear relationships.

## Illustration: Simple Neural Network Example

- **Network Setup**:
  - Input: `x` (a scalar)
  - One hidden layer: weight `w1`, bias `b1`, output `a1`
  - Output layer: weight `w2`, bias `b2`, output `a2`

- **If using linear activation (`g(z) = z`)**:
  - a1 = w1 * x + b1
  - a2 = w2 * a1 + b2 = w2 * (w1 * x + b1) + b2
  - Simplifies to: a2 = (w2 * w1) * x + (w2 * b1 + b2)
  - This is a linear function: `a2 = w * x + b`

> This proves that stacking layers with linear activation doesn’t increase expressiveness — the model is still equivalent to simple **linear regression**.

## Generalization to Multi-layer Networks

- **Multiple Hidden Layers with Linear Activation**:
  - Output is still a **linear function** of input `x`.
  - All intermediate transformations are linear → total transformation is also linear.

- **If Output Layer Uses Sigmoid, but Hidden Layers are Linear**:
  - Becomes equivalent to **logistic regression**.
  - Output: `a = 1 / (1 + e^-(w * x + b))` for some `w`, `b`.

> A linear function of a linear function is still a linear function. Without non-linearity, the depth of the network doesn't matter.

## Key Takeaways

- **Don't use linear activation functions in hidden layers**.
  - They make the network no more powerful than linear/logistic regression.
- **Use non-linear activation functions like ReLU** in hidden layers.
  - They allow the network to model complex functions and learn non-linear decision boundaries.

## Recommendation

| Layer Type      | Recommended Activation      |
|----------------|-----------------------------|
| Hidden Layers   | ReLU (default)              |
| Output (Binary Classification) | Sigmoid     |
| Output (Regression) | Linear or ReLU (if y ≥ 0) |

---
# Multiclass Classification in Neural Networks

## Multiclass Classification

- **Definition**:
  - A classification problem where the target label `y` can take on **more than two discrete values**.
  - Still a classification task — `y` is **categorical**, not continuous.
  
- **Examples**:
  - **Digit Recognition**:
    - Classifying handwritten digits (0–9), e.g., reading zip codes.
  - **Medical Diagnosis**:
    - Classifying a patient into 1 of 3–5 possible diseases.
  - **Manufacturing Defect Detection**:
    - Detecting types of defects in product images: scratch, discoloration, chip, etc.

## Binary vs Multiclass Classification

| Type                 | `y` Values           | Example                                      |
|----------------------|----------------------|----------------------------------------------|
| Binary Classification| {0, 1}               | Spam vs not spam                             |
| Multiclass Classification | {0, 1, 2, ..., K} | Digit recognition (0–9), disease types, etc. |

## Visual Example of Data

- **Binary Classification**:
  - Logistic regression estimates: `P(y = 1 | x)`
  - Data points classified into 2 categories (e.g., circles and crosses)

- **Multiclass Classification**:
  - Now estimate: `P(y = 0 | x), P(y = 1 | x), ..., P(y = K | x)`
  - Data may have multiple classes (e.g., circles, triangles, squares)
  - Algorithm learns **multi-region decision boundaries** to divide input space

## Next Steps

- **Softmax Regression**:
  - Generalization of logistic regression to handle **multiple classes**.
  - Learns to predict a **probability distribution** over all possible classes.
  
- **Neural Networks for Multiclass Classification**:
  - Use softmax in the **output layer**.
  - Train neural networks to handle more than two classes.

> Multiclass classification extends binary classification by estimating probabilities across multiple discrete classes, and softmax regression is the key model enabling this in both traditional ML and neural networks.
---
# Softmax Regression: Generalizing Logistic Regression for Multiclass Classification

## Recap: Logistic Regression

- **Binary Classification**:
  - Output label `y ∈ {0, 1}`
  - Compute:
    - `z = w · x + b`
    - `a = sigmoid(z)` → estimated `P(y = 1 | x)`
  - `P(y = 0 | x) = 1 - a`
  - Can be seen as computing:
    - `a1 = P(y = 1 | x)`
    - `a2 = P(y = 0 | x) = 1 - a1`

## Generalization to Softmax Regression

- **Multiclass Case**:
  - Output label `y ∈ {1, 2, ..., n}`
  - For each class `j`, compute:
    - `z_j = w_j · x + b_j`
  - Then compute:
    - `a_j = e^{z_j} / (e^{z_1} + e^{z_2} + ... + e^{z_n})`
    - `a_j` = estimated `P(y = j | x)`
  - All `a_j` values add up to **1** by construction

### Example with 4 classes:

- Compute:
  - `z₁ = w₁ · x + b₁`
  - `z₂ = w₂ · x + b₂`
  - `z₃ = w₃ · x + b₃`
  - `z₄ = w₄ · x + b₄`
- Compute activations:
  - `a₁ = e^{z₁} / (e^{z₁} + e^{z₂} + e^{z₃} + e^{z₄})`
  - `a₂ = e^{z₂} / (e^{z₁} + e^{z₂} + e^{z₃} + e^{z₄})`
  - `a₃ = e^{z₃} / (e^{z₁} + e^{z₂} + e^{z₃} + e^{z₄})`
  - `a₄ = e^{z₄} / (e^{z₁} + e^{z₂} + e^{z₃} + e^{z₄})`



## General Case (n classes)

- **Formulas**:
  - `z_j = w_j · x + b_j` for `j = 1 to n`
  - `a_j = e^{z_j} / sum_{k=1 to n} e^{z_k}`
  - `a_j` = `P(y = j | x)`
  - Total `a_j` values always add to **1**

- **Special Case**:
  - If `n = 2`, softmax regression reduces to logistic regression (different parameterization but same logic)

## Loss Function for Softmax Regression

- **For Logistic Regression**:
  - `Loss = -[y * log(a1) + (1 - y) * log(a2)]`
  - Since `a2 = 1 - a1`, simplified as:
    - `Loss = -log(a1)` if `y = 1`
    - `Loss = -log(a2)` if `y = 0`

- **For Softmax Regression**:
  - If `y = j`, then:
    - `Loss = -log(a_j)`
  - Encourages high confidence (i.e. `a_j` close to 1) for correct class
  - **Only one `a_j` term used per training sample** based on true label

- **Interpretation**:
  - The lower `a_j` is (i.e., less confidence), the higher the loss
  - Minimizing this loss pushes the model to increase confidence in the correct class

## Training Objective

- **Cost Function**:
  - Average of losses over all training examples
  - Minimizing cost helps find optimal parameters: `w_1...w_n`, `b_1...b_n`



- **Softmax Regression**:
  - Generalizes logistic regression to handle multiple output classes
  - Computes probability for each class
  - Uses softmax function to normalize outputs
  - Loss based on negative log likelihood of true class
  - Forms foundation for multiclass classification using neural networks

---

# Neural Network with Softmax Output Layer for Multiclass Classification

- **Softmax in Neural Networks**  
  Softmax regression is used as the output layer in neural networks to handle multi-class classification problems.

- **Binary vs. Multiclass Neural Network**  
  Previously for binary classification (e.g., digit recognition with only two classes), the network had a single output unit.  
  For multiclass classification (e.g., recognizing digits 0-9), the network needs **10 output units**—one for each class.

- **Softmax Output Layer**  
  The final layer becomes a **Softmax layer**, which outputs the probability distribution over all classes.  
  - It is often referred to as the **Softmax activation function**.

- **Forward Propagation in Multiclass NN**  
  1. Input `x` is passed through hidden layers just like before.
  2. Final layer computes `z₁` to `z₁₀` using:  
     `z_j = w_j ⋅ a² + b_j` (where `a²` is the activation from the previous layer)
  3. Output activations `a₁` to `a₁₀` are computed using the softmax formula:  
     `a_j = e^(z_j) / Σₖ e^(z_k)` for k from 1 to 10  
     These represent probabilities of each class.

- **Layer Notation**  
  - Outputs of the final layer are sometimes labeled with superscripts to indicate the layer, e.g., `z₁^(3)`, `a₁^(3)`.

- **Unique Property of Softmax**  
  - Unlike other activations (ReLU, sigmoid, etc.), **softmax activations are interdependent**:  
    Each output `a_j` depends on *all* `z` values in that layer (z₁ to z₁₀), not just its corresponding `z_j`.

- **TensorFlow Implementation Overview**  
  To build such a model in TensorFlow, follow three steps:
  1. **Define Model Layers:**
     - Input layer → Dense(25, activation='relu')
     - Hidden layer → Dense(15, activation='relu')
     - Output layer → Dense(10, activation='softmax')
  2. **Loss Function:**
     - Use `SparseCategoricalCrossentropy` for multiclass classification.
     - "Sparse" means y takes only one of the n class values (not multi-label).
  3. **Model Training:**
     - Same training flow as binary classification models (e.g., compile, fit, etc.)



- **Terminology**  
  - **Sparse Categorical**: y takes one value from a set of discrete categories.
  - **Softmax Activation**: Produces a normalized probability distribution across multiple classes.

---
# Numerically Stable Softmax and Logistic Regression in TensorFlow

## Numerical Accuracy in Computation
- **Floating-point Precision**: Computers store numbers using a finite amount of memory (floating-point numbers), which leads to *round-off errors* in computations.
- **Two Ways to Compute 2/10,000**:
  - Option 1: `x = 2 / 10000` (Direct and accurate).
  - Option 2: `x = (1 + 1/10000) - (1 - 1/10000)` (Equivalent in theory but more prone to round-off errors due to floating-point precision).

## Problem with Intermediate Computations
- **Computing Intermediate Terms**: Insisting on computing intermediate values like `a = 1 / (1 + e^-z)` can cause accumulation of numerical errors.
- **Rewriting Expressions**: Rewriting loss functions to avoid intermediate computations allows TensorFlow to perform internal rearrangements for better numerical stability.

## Logistic Regression Example
- **Typical Loss Computation**:
  - Compute activation: `a = 1 / (1 + e^-z)`
  - Compute loss: `-y * log(a) - (1 - y) * log(1 - a)`
- **Better Practice**:
  - Skip computing `a` explicitly.
  - Provide the expanded expression directly to the loss function.
  - Use TensorFlow’s built-in functions with `from_logits=True`.

## TensorFlow Implementation Tip
- **Using `from_logits=True`**:
  - When defining the loss (e.g., `BinaryCrossentropy(from_logits=True)`), TensorFlow treats model outputs as raw logits (z values).
  - This lets TensorFlow compute the sigmoid or softmax internally in a more stable way.

## Multiclass Classification with Softmax
- **Traditional Softmax Computation**:
  - Compute `a_j = e^z_j / Σ e^z_k`
  - Compute loss: `-log(a_y)`
  - This is split into two steps and may suffer from round-off errors when `z` values are very large or small.
- **Improved Implementation**:
  - Let model output logits `z_1 to z_10` directly using a *linear activation*.
  - Use `SparseCategoricalCrossentropy(from_logits=True)` to let TensorFlow handle softmax + cross-entropy internally.
- **Numerical Stability**:
  - TensorFlow avoids computing large exponentials directly.
  - Leads to more accurate loss computations.

## Recommended Practices
- **Avoid Explicit Activation Functions in Final Layer for Loss Calculation**:
  - Use linear activation for the output layer.
  - Use the `from_logits=True` parameter in loss functions.
- **Trade-off**:
  - Slightly less readable code.
  - Significantly more numerically accurate computations.
- **Probabilities**:
  - If needed, apply softmax manually afterward to get actual probability outputs from logits.


- **Multi-label Classification**:
  - Mentioned briefly as the next topic.
  - Differs from multiclass classification — one input can belong to multiple classes.

---












































