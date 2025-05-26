# Sequence Models 

## Introduction

- **Importance**: Sequence models, especially **Recurrent Neural Networks (RNNs)** and their variants, have revolutionized fields like:
  - Speech recognition
  - Natural language processing (NLP)
  - Music generation
  - DNA sequence analysis
  - Machine translation
  - Video activity recognition
  - Named entity recognition (NER)

## Sequence Model Applications

### 1. **Speech Recognition**
- **Input (X)**: Audio clip (sequence over time)
- **Output (Y)**: Text transcript (sequence of words)
- **Both input and output are sequences**

### 2. **Music Generation**
- **Input (X)**: Could be empty or a single integer (e.g., music genre or a few starting notes)
- **Output (Y)**: Sequence of musical notes
- **Only output is a sequence**

### 3. **Sentiment Classification**
- **Input (X)**: Sequence of words (text review)
- **Output (Y)**: Single label (e.g., star rating)
- **Only input is a sequence**

### 4. **DNA Sequence Analysis**
- **Input (X)**: DNA sequence (letters A, C, G, T)
- **Output (Y)**: Label indicating functional regions (e.g., protein-coding)
- **Only input is a sequence**

### 5. **Machine Translation**
- **Input (X)**: Sentence in one language
- **Output (Y)**: Translated sentence in another language
- **Both input and output are sequences** (possibly of different lengths)

### 6. **Video Activity Recognition**
- **Input (X)**: Sequence of video frames
- **Output (Y)**: Activity label
- **Only input is a sequence**

### 7. **Named Entity Recognition (NER)**
- **Input (X)**: Sentence
- **Output (Y)**: Labels identifying entities (e.g., person names)
- **Input and output are sequences of the same length**

## Supervised Learning Context
- **All tasks** can be framed as **supervised learning** with labeled data pairs: `(X, Y)`
- Sequence models are trained using labeled examples from the specific domain.

## Sequence Types
- **X and Y both sequences**: Speech recognition, machine translation, NER
- **Only X is a sequence**: Sentiment classification, DNA analysis, video recognition
- **Only Y is a sequence**: Music generation
- **Different lengths**: Input and output sequences can have varying or same lengths depending on the task


- **Versatility**: Sequence models handle diverse and complex tasks involving sequential data.
---

# Notation and Word Representation 


## Named Entity Recognition (NER)
- **Task**: Given a sentence, output a label for each word identifying whether it's part of a person’s name.
- **Practical Use**: Used by search engines to index names of people, companies, locations, dates, currencies, etc.

## Input and Output Structure
- **Input (X)**: Sequence of words, e.g., “Harry Potter and Hermione Granger invented a new spell.”
  - Number of words = 9 → `T_x = 9`
- **Output (Y)**: Sequence of labels corresponding to input words.
  - For this example: also 9 labels → `T_y = 9`
  - Output can simply be binary (e.g., 1 = person name, 0 = not a person name)

## Notation Summary
- `x⟨t⟩`: t-th word in the input sequence
- `y⟨t⟩`: t-th label/output corresponding to `x⟨t⟩`
- `T_x`: Length of input sequence
- `T_y`: Length of output sequence
- For multiple training examples:
  - `x^(i)⟨t⟩`: t-th word in input sequence of i-th training example
  - `T_x^(i)`: Input sequence length for example i
  - `y^(i)⟨t⟩`: t-th output for training example i
  - `T_y^(i)`: Output sequence length for example i

## Vocabulary (Dictionary)
- **Vocabulary**: A list of all possible words to be used in input representation.
  - Example words and indices:
    - "a" → index 1
    - "Aaron" → index 2
    - "and" → index 367
    - "Harry" → index 4075
    - "Potter" → index 6830
    - "Zulu" → index 10,000
- **Vocabulary Size**:
  - For illustration: 10,000 words
  - Real applications: typically 30,000–100,000 words
  - Some large-scale models use vocabularies of 1 million+ words

## Word Representation: One-Hot Encoding
- **One-hot vector**: Vector of all 0s with a single 1 at the index of the word in the vocabulary.
  - If vocabulary size = 10,000 → each word is a 10,000-dimensional vector
  - Example:
    - "Harry" (index 4075) → vector with 1 at position 4075
    - "Potter" (index 6830) → 1 at 6830
    - "and" (index 367) → 1 at 367
- **Each `x⟨t⟩`** is represented by a one-hot vector
- **Each sentence** is represented by a sequence of one-hot vectors (one per word)

## Handling Unknown Words
- **Problem**: A word not in the vocabulary appears during inference.
- **Solution**: Use a special token: `UNK` (Unknown Word) to represent all out-of-vocabulary words

---

# Recurrent Neural Networks (RNNs) – Summary Notes

## Problem with Standard Neural Networks for Sequence Learning
- **Variable Input/Output Lengths**: 
  - Standard neural networks assume fixed-size inputs/outputs.
  - In sequence data (e.g., sentences), input length (Tx) and output length (Ty) can vary.
- **No Feature Sharing**: 
  - A standard NN learns separate features for each word position.
  - Doesn’t generalize well across different positions in the sequence.
- **High Dimensionality**:
  - One-hot encodings lead to large input sizes (e.g., 9 words × 10,000-dim vectors).
  - Results in a huge number of parameters in weight matrices.

## Introduction to Recurrent Neural Networks (RNNs)
- **Sequential Processing**: 
  - Processes input one timestep at a time (e.g., X1, X2, ..., X_Tx).
  - Maintains an internal hidden state across timesteps.
- **Information Flow**: 
  - Hidden state from timestep t-1 is used in timestep t.
  - Allows the model to carry context forward in the sequence.

## RNN Architecture
- **Unrolling the RNN**:
  - Visualize RNN as a series of repeated cells (one per timestep).
  - Each cell shares the same parameters across time.
- **Input and Output at Each Timestep**:
  - Input: x<t>
  - Hidden state: a<t>
  - Output: ŷ<t>

## Parameter Sharing
- **Parameters Used**:
  - `Waa`: weights for previous activation `a<t-1>` to current activation.
  - `Wax`: weights for input `x<t>` to activation.
  - `Wya`: weights from activation to output.
  - `ba`: bias for activation computation.
  - `by`: bias for output computation.
- **All parameters are shared** across all timesteps (reduces number of parameters).

## Forward Propagation in RNN
- **Initialization**:
  - `a<0>` is usually a zero vector (can also be random).
- **Equations**:
  - Hidden state:  
    `a<t> = g(Waa * a<t-1> + Wax * x<t> + ba)`
  - Output:  
    `ŷ<t> = g(Wya * a<t> + by)`
- **Activation Functions**:
  - `tanh` is common for computing `a<t>`.
  - `sigmoid` or `softmax` used depending on output type (e.g., binary or multiclass).

## Compact Notation (Simplified Equations)
- **Combined Weight Matrix**:
  - Define `Wa = [Waa | Wax]` by stacking horizontally.
  - Input vector: `[a<t-1>; x<t>]` (concatenation of previous hidden state and input).
- **Updated Equations**:
  - `a<t> = g(Wa * [a<t-1>; x<t>] + ba)`
  - `ŷ<t> = g(Wy * a<t> + by)`

## Limitations of Unidirectional RNN
- **One-Sided Context**:
  - Only uses past information (left-to-right) for prediction.
  - Cannot leverage future context (e.g., to disambiguate "Teddy").




- RNNs are effective for variable-length sequences and shared-feature learning.
- Forward propagation involves computing activations and outputs at each timestep using shared parameters.
- Compact notation simplifies the implementation of more complex models.
- Limitation: only past information is used; solved later using Bi-Directional RNNs.

---

# Backpropagation in Recurrent Neural Networks (RNNs)

##  Forward Propagation Recap
- **Sequence Input:** Input sequence consists of tokens \( x_1, x_2, ..., x_{T_x} \).
- **Activation Computation:** At each timestep \( t \), activation \( a_t \) is computed using:
  - Previous activation \( a_{t-1} \)
  - Current input \( x_t \)
  - Shared parameters \( W_a \), \( b_a \)
- **Prediction Output:** At each timestep, prediction \( \hat{y}_t \) is computed using:
  - Current activation \( a_t \)
  - Shared parameters \( W_y \), \( b_y \)
- **Parameter Sharing:** Same weights \( W_a, b_a \) and \( W_y, b_y \) are used at all timesteps.

##  Loss Function
- **Element-wise Loss:** Cross-entropy loss (logistic loss) for each prediction:
  ```math
  \mathcal{L}(\hat{y}_t, y_t) = -y_t \log(\hat{y}_t) - (1 - y_t) \log(1 - \hat{y}_t)
  ```
- **Total Loss:** Sum of losses across all timesteps:
  ```math
  \mathcal{L} = \sum_{t=1}^{T_x} \mathcal{L}(\hat{y}_t, y_t)
  ```

##  Backpropagation Through Time (BPTT)
- **Direction:** Reverse of forward propagation (right to left across time).
- **Goal:** Compute gradients of all parameters \( W_a, b_a, W_y, b_y \).
- **Backprop Mechanics:** Use chain rule across timesteps to propagate error signals.
- **Gradient Flow:** The most critical recursive computation is across time (horizontal connections).
- **Update Step:** Use computed gradients to update parameters via gradient descent.

---

# RNN Architectures for Different Input/Output Sequence Lengths



## Common RNN Architectures

### 1. **One-to-One**
- **Description:** Basic feedforward neural network.
- **Use Case:** Simple tasks with a single input and output.
- **Example:** Basic classification, e.g., predicting a label from a fixed-size feature vector.

### 2. **One-to-Many**
- **Description:** Single input leading to a sequence of outputs.
- **Use Case:** Sequence generation.
- **Example:** Music generation, where the input might be a genre or starting note, and the output is a sequence of musical notes.

### 3. **Many-to-One**
- **Description:** Sequence input, single output.
- **Use Case:** Classification tasks based on sequential data.
- **Example:** Sentiment classification of movie reviews where the input is a sentence and the output is a rating or sentiment label.

### 4. **Many-to-Many (Equal Length)**
- **Description:** Input and output are sequences of the same length.
- **Use Case:** Labeling each input timestep with a prediction.
- **Example:** Named Entity Recognition (NER), where each word in a sentence is labeled.

### 5. **Many-to-Many (Different Length)**
- **Description:** Input and output are both sequences but of different lengths.
- **Use Case:** Tasks where input and output sequence lengths vary.
- **Example:** Machine translation (e.g., French to English translation).
  - **Architecture:**
    - **Encoder:** Processes the input sequence.
    - **Decoder:** Generates the output sequence from the encoded context.

## Summary Table of Architectures

| Architecture       | Input Type     | Output Type    | Example                          |
|--------------------|----------------|----------------|----------------------------------|
| One-to-One         | Single input   | Single output  | Basic binary classification      |
| One-to-Many        | Single input   | Sequence output| Music or text generation         |
| Many-to-One        | Sequence input | Single output  | Sentiment classification         |
| Many-to-Many (Equal)| Sequence input | Sequence output| Named Entity Recognition (NER)   |
| Many-to-Many (Diff)| Sequence input | Sequence output| Machine translation              |


- **Parameter Sharing:** RNN parameters (weights/biases) are shared across all timesteps.
- **Sequence Generation Nuance:** In generation tasks, synthesized outputs are often fed back as inputs for the next timestep.
- **Encoder-Decoder Framework:** Crucial for tasks like machine translation where input and output lengths differ.


- The basic RNN building blocks can be rearranged to create a wide variety of architectures suitable for diverse NLP and sequence modeling problems.
- These flexible designs make RNNs powerful tools for processing and generating sequential data.
---

- RNN language models learn to predict the next word in a sequence.
- Can compute the probability of entire sentences.
- Training involves minimizing the sum of cross-entropy losses over time steps.
- Forms the basis for sequence generation (covered next).

---