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

# Recurrent Neural Networks (RNNs)

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

# Gated Recurrent Unit (GRU) 


- **Standard RNN Equation**:
  ```math  
   a^{\langle t \rangle} = \tanh(W_a [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b) 
  ```
  
  - Inputs: $ a^{\langle t-1 \rangle} $, $ x^{\langle t \rangle} $  
  - Output: $ a^{\langle t \rangle} $ → can be passed to a softmax for prediction

##  Motivation for GRUs
- RNNs struggle with long-range dependencies due to vanishing gradients.
- GRUs introduce **gating mechanisms** to solve this problem.
- GRUs help **retain important information** (e.g., subject-verb agreement like "the cat was").

---

##  GRU Structure Overview
- Adds a **memory cell** $ c^{\langle t \rangle} $, which also equals the output $ a^{\langle t \rangle} $ in GRUs.
- This memory can persist relevant info across time steps.
- GRUs decide **when to update** or **retain memory** via gates.

---

##  Update Mechanism
- **Candidate Memory Cell**:
  ```math  
   \tilde{c}^{\langle t \rangle} = \tanh(W_c [\Gamma_r * c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)
  ```
- **Update Gate (Γ_u)**:
  ```math  
   \Gamma_u = \sigma(W_u [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u)
  ```  
  - Values between 0 and 1 (via sigmoid)
  - Intuition: 
    - $ \Gamma_u = 1 $ → update memory  
    - $ \Gamma_u = 0 $ → retain old memory
- **Memory Update Equation**: 
  ```math
   c^{\langle t \rangle} = \Gamma_u * \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) * c^{\langle t-1 \rangle}
  ```

---

##  Relevance Gate (Full GRU)
- Adds an **additional gate** $ \Gamma_r $ (Relevance gate):  
  - Controls how much of the past cell $ c^{\langle t-1 \rangle} $ is used to compute $ \tilde{c}^{\langle t \rangle} $
- **Relevance Gate Equation**:
  ```math  
   \Gamma_r = \sigma(W_r [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_r) 
  ```

---

##  Element-wise Operations
- All operations are **element-wise** (denoted by \( * \)):
  - $ c^{\langle t \rangle}, \Gamma_u, \tilde{c}^{\langle t \rangle}, \Gamma_r $ are vectors (e.g., 100-dimensional)
  - Each bit/dimension can be selectively updated or retained

---


- GRUs retain long-term dependencies by **remembering key bits** and only updating when needed.
- Helps solve **vanishing gradient** by ensuring values can be passed along unchanged.
- Gates can be nearly exactly 0 or 1 → near-perfect memory preservation.

---


- Inputs: $ c^{\langle t-1 \rangle} (= a^{\langle t-1 \rangle}), x^{\langle t \rangle} $
- Internal components:
  - Compute $ \tilde{c}^{\langle t \rangle} $ via tanh
  - Compute $ \Gamma_u $ via sigmoid
  - Optional: compute $ \Gamma_r $ for full GRU
- Combine all in final equation to get $ c^{\langle t \rangle} (= a^{\langle t \rangle}) $
- May feed into softmax for prediction $ \hat{y}^{\langle t \rangle} $

---

# LSTM (Long Short-Term Memory) Summary 

- Developed by Sepp Hochreiter & Jürgen Schmidhuber.
- More powerful and general than GRU.
- Introduces separate **memory cell** (`cₜ`) and **hidden state** (`aₜ`), unlike GRU where `aₜ = cₜ`.

##  LSTM Gates
- Uses three gates (vs GRU’s two):
  - **Forget Gate (γ_f)**: decides what to discard from `cₜ₋₁`.
  - **Update Gate (γ_u)**: decides how much new info to add from `c̃ₜ`.
  - **Output Gate (γ_o)**: decides what part of memory to output as `aₜ`.

##  LSTM Equations
- **Forget Gate**:  
  `γ_f = σ(W_f [aₜ₋₁, xₜ] + b_f)`
- **Update Gate**:  
  `γ_u = σ(W_u [aₜ₋₁, xₜ] + b_u)`
- **Candidate Cell Value**:  
  `c̃ₜ = tanh(W_c [aₜ₋₁, xₜ] + b_c)`
- **Memory Cell Update**:  
  `cₜ = γ_u ⊙ c̃ₜ + γ_f ⊙ cₜ₋₁`
- **Output Gate**:  
  `γ_o = σ(W_o [aₜ₋₁, xₜ] + b_o)`
- **Hidden State**:  
  `aₜ = γ_o ⊙ tanh(cₜ)`
- `⊙` denotes element-wise multiplication.

---
![lstmm](https://github.com/user-attachments/assets/fe4d267b-3292-4d82-b127-0267b7099b90)

- All gates computed from `aₜ₋₁` and `xₜ`.
- These values (γ_f, γ_u, γ_o, c̃ₜ) are combined to update `cₜ` and generate `aₜ`.
- Memory `cₜ` can retain values across many time steps if gates are set appropriately.
- Enables **long-term memory** without vanishing gradients.

## Peephole Connections
- Variation: gates also depend on `cₜ₋₁`.
  - Gate inputs: `[aₜ₋₁, xₜ, cₜ₋₁]`
  - Known as **peephole connections**.
- Peephole connections apply element-wise:
  - i-th element of `cₜ₋₁` only affects i-th element of the gate.

## When to Use GRU vs. LSTM
- **LSTM**
  - More powerful and flexible.
  - Better for tasks with complex long-term dependencies.
  - Historically more widely used and tested.
- **GRU**
  - Simpler, faster, fewer parameters.
  - Easier to train larger models.
  - Often performs as well as LSTM in practice.

---
- Both LSTM and GRU are effective for sequence modeling.
- LSTM is default choice for complex tasks.
- GRU is preferred for simpler, faster models or when computational efficiency is critical.

---

# Bidirectional RNN (BRNN) 



##  Motivation for BRNN
[brnn1]
- **Example: Named Entity Recognition (NER)**:
  - Standard (unidirectional) RNNs may fail to determine if a word like *“Teddy”* is part of a name or just refers to a teddy bear.
  - Knowing *future* words like *“Roosevelt”* helps determine context.
  - This limitation exists regardless of using vanilla RNN, GRU, or LSTM units.

---

##  How BRNN Works
- **Inputs**: $ x_1, x_2, x_3, x_4 $
- **Forward Pass**:
  - Computes hidden states from left to right: $ \vec{a}_1, \vec{a}_2, \vec{a}_3, \vec{a}_4 $
- **Backward Pass**:
  - Computes hidden states from right to left: $ \vec{a}_4, \vec{a}_3, \vec{a}_2, \vec{a}_1 $
- **Connections**:
  - Forward hidden states are connected to each other going forward.
  - Backward hidden states are connected in reverse order.

---

##  Making Predictions
- **At each time step \( t \)**:
  - Output prediction $\hat{y}^{(t)} $ uses both forward and backward activations:
  ```math
    
    \hat{y}^{(t)} = g(W_y[\vec{a}^{(t)}; \vec{a}^{(t)}] + b_y)
    
  ```
  - This enables the model to utilize both **past** (via forward states) and **future** (via backward states) information.

---

## Example 
[bnnn2]
- To classify the word "Teddy":
  - Forward pass brings in context from “He said Teddy”.
  - Backward pass brings in context from “Roosevelt”.
  - Combines both directions to accurately classify as a person’s name.

---

## Units Used in BRNNs
- BRNNs can use:
  - **Vanilla RNN units**
  - **GRU units**
  - **LSTM units**
-  Commonly, **LSTM-based BRNNs** are used in NLP tasks.

---

##  Applications
- **Effective for NLP tasks** where the entire input sequence (like a sentence) is available at once:
  - Named Entity Recognition (NER)
  - Part-of-Speech (POS) tagging
  - Sentiment analysis
-  LSTM-based BRNNs are a **strong first choice** for such tasks.

---

##  Limitation
- BRNNs require access to the **entire input sequence** before making predictions.
  - Not suitable for **real-time** applications (e.g., live speech recognition).
  - Must wait until the full input is available.

---

# Deep RNN 

##  Motivation for Deep RNNs
- **Standard RNNs (or GRUs/LSTMs)** are already effective.
- But for learning **more complex functions**, it can help to **stack multiple RNN layers**—this creates a **Deep RNN**.

---

##  Comparison to Deep Feedforward Networks
- In a feedforward neural network:
  - Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output
- Similarly, in a deep RNN:
  - RNN layers are **stacked vertically** (depth), and **unrolled in time** (temporally).
  - Each layer processes the sequence and passes its outputs to the next layer above.

---

##  Notation
- **Activation notation**:
  - $ a^{[l]\langle t \rangle} $: Activation at layer `l` and time step `t`.
- Each RNN layer has its **own parameters**:
  - Layer 1: $ W_a^{[1]}, b_a^{[1]} $
  - Layer 2: $ W_a^{[2]}, b_a^{[2]} $
- Hidden states depend on:
  - **Previous time step** (horizontal input)
  - **Lower layer output** at current time step (vertical input)

---

##  Activation Example
- To compute $ a^{[2]\langle 3 \rangle} $:
  - Use:
    - Horizontal input: $ a^{[2]\langle 2 \rangle} $
    - Vertical input: $ a^{[1]\langle 3 \rangle} $
  - Compute using:
  ```math
    a^{[2]\langle 3 \rangle} = g(W_a^{[2]}[a^{[2]\langle 2 \rangle}, a^{[1]\langle 3 \rangle}] + b_a^{[2]})
  ```

---

##  Architecture Variants
- **Fully stacked deep RNN**:
  - Each layer is connected **both temporally and vertically**.
  - Typically only **a few layers deep** (e.g., 2–3 layers) due to temporal complexity.
- **Recurrent + Deep Feedforward**:
  - Recurrent layers are followed by **deep feedforward networks**.
  - These feedforward networks make predictions at each time step.
  - These additional layers are **not connected temporally** (no horizontal connections).

---

##  Types of Units
- Each layer in the deep RNN can use:
  - **Standard RNN units**
  - **GRU units**
  - **LSTM units**

---

##  Deep Bidirectional RNNs
- It’s possible to stack **Bidirectional RNNs** too.
- You can have multiple **forward + backward layers** stacked vertically.
- However, these are **computationally expensive** and rare in practice.

---


- RNNs with **3 stacked recurrent layers** are already considered "deep".
- **Temporal dimension** makes RNNs inherently complex.
- In contrast, feedforward networks can go **100+ layers deep**.

---

# Word Embeddings in NLP



##  One-Hot Representation: Limitations
- Words are represented using **one-hot vectors**: binary vectors with a single `1` in the position corresponding to the word.
  - Example: If “man” is word 5391 in vocabulary → $ O_{5391} $
- Problems:
  - Treats all words as **completely unrelated**.
  - **Dot product** and **distance** between any two different one-hot vectors is always zero or same.
  - Hard for model to generalize: e.g., knowing "orange juice" doesn’t help it understand "apple juice".
[0-h vec]
---

## Embedding Representation: Motivation
- Replace one-hot vectors with **feature-rich vectors** (embeddings).
- Represent words with **dense vectors** of, say, 300 real-valued numbers.
  - Example: $e_{5391} $ for “man” instead of $ O_{5391} $
- Embedding features could capture:
  - **Gender**: -1 (male) to +1 (female)
  - **Royalty**: e.g., king/queen > man/woman
  - **Age**, **food type**, **living vs non-living**, **noun/verb**, etc.

---

## Benefits of Embeddings
- Helps model generalize:
  - Similar words (like “apple” and “orange”) have **similar embeddings**.
  - Improves predictions like “apple juice” when model knows “orange juice”.
- Embeddings capture **semantic similarity**:
  - “Man” and “woman” are closer in embedding space than “man” and “orange”.

---

## Learning Embeddings
[learningembedding]
- Embeddings are **learned** from data, not manually engineered.
- Though we might imagine intuitive features (e.g., gender), the **actual vector dimensions** may not be directly interpretable.

---

## Visualization

![t-sne](https://github.com/user-attachments/assets/565d37d3-b2f4-42ce-85ba-37b11f0e4c42)

- High-dimensional embeddings (e.g., 300D) can be **projected to 2D** using algorithms like **t-SNE** (by van der Maaten & Hinton).
- Visualizations show:
  - Similar words cluster together (e.g., fruits, animals, people).
  - Related categories form **distinct groups** in lower-dimensional space.

---

##  Embedding Terminology

- Each word is **embedded** as a point in this high-dimensional space.
- Useful for understanding relationships and enabling downstream tasks (e.g., sentiment analysis, translation).

---


- **One-hot encodings** are sparse and uninformative for word similarity.
- **Word embeddings** provide compact, meaningful vector representations.
- These allow models to **generalize**, understand **analogies**, and perform better in NLP tasks.
- Visualization (e.g., via t-SNE) confirms the grouping of semantically similar words.
- Word embeddings are foundational to modern NLP systems and enable more **data-efficient** learning.

---


# Word Embeddings and NLP Applications


- **Featurized Representations**: Embedding vectors provide richer representations than one-hot encodings, capturing semantic similarity.
- **Named Entity Recognition Example**: Detect names like "Sally Johnson" or "Robert Lin" by leveraging context ("orange/apple farmer").

## Generalization via Embeddings
- **Generalizing to Unseen Words**:
  - Example: “Robert Lin is a durian cultivator” may contain unseen words (“durian”, “cultivator”).
  - Embeddings help generalize by knowing durian ≈ fruit and cultivator ≈ farmer.
- **Benefit**: Even with small labeled training sets, embeddings trained on large corpora help models recognize patterns with new words.

## Training on Large Corpora
- **Unlabeled Text**: Embeddings are trained using large datasets (e.g., 1–100 billion words) scraped from the internet.
- **Semantic Similarity**: Algorithms learn similarity between words like “apple” and “orange”, or “farmer” and “cultivator”.

## Transfer Learning with Embeddings
- **Pretrained Embeddings**: Use embeddings trained on large corpora and apply to tasks with small labeled datasets.
- **Example Tasks**: Named Entity Recognition, Text Summarization, Co-reference Resolution, Parsing.
- **Embedding Representation**: Words represented as 300-dimensional dense vectors instead of sparse 10,000-dim one-hot vectors.

## Fine-tuning Embeddings
- **Optional Fine-tuning**:
  - Fine-tuning the embeddings on labeled dataset can improve performance **if** the dataset is sufficiently large.
  - If the labeled dataset is small, don’t fine-tune—just use pretrained embeddings as-is.

## Embeddings Are Most Useful When:
- **Labelled Dataset is Small**: Embeddings significantly boost performance when data for the target task is limited.
- **Less Useful For**:
  - Language modeling
  - Machine translation (if large task-specific datasets exist)

##  Word Embeddings vs Face Encodings
- **Similarity**: Both represent high-dimensional dense features to capture similarity.
- **Face Encodings**:
  - Neural networks compute encodings for any new image.
- **Word Embeddings**:
  - Embeddings are **precomputed** for a fixed vocabulary (e.g., 10,000 words).
  - Unknown words are marked as “UNK”.

## Transfer Learning Using Embeddings
1. **Train word embeddings** from large unlabeled text corpora (or use pretrained).
2. **Apply embeddings** to a new NLP task with smaller labeled datasets.
3. **Optionally fine-tune** embeddings during training if data allows.
4. **Replace one-hot vectors** with dense, lower-dimensional embedding vectors.

---
# Word Embeddings for Analogy Reasoning

## Vector Arithmetic for Analogies
- **Featurized Representations**  
  Each word is represented as a dense vector (typically 50–1000 dimensions, simplified to 4D in examples).
  
- **Vector Difference Property**  
  - Difference between vectors (e.g., `e_man - e_woman`) captures meaningful relations like gender.
  - Similarly, `e_king - e_queen` also captures a gender-based relation.
  - These differences are often nearly equal: `e_man - e_woman ≈ e_king - e_queen`.

## Analogy Reasoning Mechanism
- **Core Equation for Analogies**  
  To solve analogies like “man is to woman as king is to ?”, compute:
  `e_w ≈ e_king - e_man + e_woman`
  Search for the word `w` whose embedding is most similar to the resulting vector.

- **Similarity Metric: Cosine Similarity**  
- Formula:  
  ```
  similarity(u, v) = (uᵀv) / (||u|| * ||v||)
  ```
- Measures the cosine of the angle between vectors.
- Values:
  - 1 → vectors point in same direction (max similarity),
  - 0 → orthogonal (no similarity),
  - -1 → opposite directions.
- Preferred over Euclidean distance due to normalization.

## Performance and Visualization
- **Effectiveness of Analogy Tasks**  
- Algorithms often achieve 30%–75% accuracy depending on data and method.
- Accuracy measured by whether exact correct word is retrieved.

- **Limitations of t-SNE Visualization**  
- t-SNE maps high-dimensional vectors to 2D for visualization.
- These mappings are non-linear, so analogy vector relationships (like parallelograms) may not visually persist.
- Analogy relationships are best analyzed in original high-dimensional space.

---
# Embedding Matrix
- A word embedding matrix `E` is learned, with each column corresponding to a word in the vocabulary.
- To retrieve a word's embedding, multiply `E` by its one-hot encoded vector.
- This simple matrix operation effectively selects the dense vector representing the target word.

---

# Language Models and Neural Networks

- Neural language models can be used to learn embeddings by predicting next words from context.
- Simpler models like Skip-Gram and CBOW also work well for learning embeddings.
- Embedding quality often depends more on training data and objective design than model complexity.

- **Skip-Gram Model**  
  Use one context word to predict nearby words — simpler but surprisingly effective.  
  Formalized in the upcoming video on Word2Ve

## Feeding into Neural Network
- **Network Structure**  
  - Input: Concatenate embeddings of all context words (e.g., 6 words × 300 dims = 1800-dimensional input).  
  - Hidden Layer: Parameterized by `W1`, `b1`.  
  - Output Layer: Softmax with parameters `W2`, `b2` to classify the next word among 10,000 possibilities.

- **Context Window**  
  More common to use a fixed-size historical window (e.g., previous 4 words) for prediction.  
  Fixed-size inputs make it easier to train on variable-length sentences.

## Parameter Sharing
- **Shared Embedding Matrix**  
  Matrix `E` is shared across all context positions — same matrix is used for each input word.

- **Training Objective**  
  Use gradient descent to maximize the likelihood of predicting the next word given context.  
  This trains both the embedding matrix `E` and the network weights (`W1`, `b1`, `W2`, `b2`).

## Intuition Behind Embedding Similarity
- **Example: Orange Juice vs. Apple Juice**  
  Model learns similar embeddings for "orange" and "apple" because both help predict "juice".  
  Embedding similarity helps the model generalize better on similar contexts.

## Generalizing the Context
- **Alternative Context Definitions**  
  - **Fixed History:** Previous four words (standard for language models).
  - **Bidirectional Context:** Use words before and after target word (e.g., 4 left + 4 right words).
  - **Single Word Contexts:** Use just one previous word.
  - **Nearby Word Context:** Use any nearby word in the sentence as context.

- **Simplification Trend**  
  Even simple context setups (like one-word context) can yield good embeddings.

---

## Skip-Gram Model Concept
- **Context & Target**:
  - Randomly choose a **context word** (e.g., "orange").
  - Select a **target word** randomly from within a window (e.g., ±5 or ±10 words around the context).
- **Supervised Learning Setup**:
  - Input: Context word.
  - Output: Target word within the window.

## Training Objective
- **Purpose**: Not to optimize prediction accuracy per se, but to learn good word embeddings from this task.
- **Vocabulary**: Typically 10,000+ words; can scale to 1M+.

## Model Architecture
- **Input**: One-hot encoded vector of the context word (e.g., `O_c` for "orange").
- **Embedding Matrix**: Multiply `E × O_c` to get embedding `E_c`.
- **Softmax Layer**:
  - Computes probability distribution over all vocabulary words as possible targets.
  - Formula:
    ```
    P(t | c) = exp(θ_tᵗ · E_c) / ∑ₖ exp(θ_kᵗ · E_c)
    ```
  - Where `θ_t` is the parameter vector for target word `t`.

## Loss Function
- **Softmax Loss**:
  - Use one-hot encoded target vector `y` and predicted probabilities `ŷ`.
  - Cross-entropy loss:
    ```
    L = -∑ y_i · log(ŷ_i)
    ```
  - Only the index corresponding to the target word is `1` in `y`; others are `0`.

## Efficiency Problem
- **Softmax Bottleneck**:
  - Computing denominator in softmax (sum over entire vocab) is slow.
  - Especially problematic for large vocabularies (100K+ words).

## Solution 1: Hierarchical Softmax
- **Binary Tree Structure**:
  - Use a tree of binary classifiers instead of flat softmax.
  - Traverse from root to leaf to classify the word.
- **Efficiency**:
  - Reduces complexity from **O(V)** to **O(log V)**.
- **Word Frequency Optimization**:
  - Frequent words (e.g., "the", "and") are placed near the top of the tree.
  - Rare words (e.g., "durian") are placed deeper in the tree.

## Word Sampling for Training
- **Context Word Sampling**:
  - Not sampled uniformly due to dominance of common words.
  - Use heuristics to balance frequency:
    - Prevent frequent updates to only high-frequency words.
    - Ensure rare words also receive training updates.

## Alternative Model: CBOW
- **CBOW (Continuous Bag of Words)**:
  - Reverses skip-gram: use context words to predict the center word.
  - Also effective with pros and cons compared to skip-gram.

---

# Negative Sampling in Skip-Gram Word Embedding





## Model Architecture

- **Binary Classification Model**:
  - For each pair `(c, t)`, model `P(y=1|c,t)` using logistic regression:
    ```
    P(y=1|c,t) = sigmoid(θ_tᵀ · e_c)
    ```
    - `θ_t`: embedding for target word `t`.
    - `e_c`: embedding for context word `c`.

- **Training Mechanism**:
  - Train on `k+1` binary classification problems per positive sample:
    - 1 positive example.
    - k negative examples.

---

## Efficient Computation

- **Softmax vs. Negative Sampling**:
  - Softmax: Expensive, requires updating all output classes (e.g., 10,000).
  - Negative Sampling:
    - Convert problem to 10,000 binary classifiers (one per word).
    - Only update `k+1` classifiers per training step (positive + negatives).
    - Dramatically reduces computation cost.

---

## Neural Network View

- **Input Layer**:
  - One-hot encode the context word.
  - Use embedding matrix to extract vector `e_c`.
- **Output**:
  - Only `k+1` logistic regression units are updated:
    - One corresponds to the actual target.
    - `k` correspond to randomly sampled words.
- **Output Layer**:
  - Think of it as 10,000 logistic units, but only a small subset are active per step.

---

## Choosing Negative Samples

- **Sampling Strategies**:
  1. **Empirical frequency**:
     - Too many common words like "the", "of", etc.
  2. **Uniform distribution**:
     - Not representative of language structure.
  3. **Best Practice**:
     - Sample according to:
       ```
       P(w_i) ∝ f(w_i)^(3/4)
       ```
     - `f(w_i)`: frequency of word `w_i` in corpus.
     - Balances between over-sampling common and rare words.
     - Empirical success, though not theoretically justified.

---
# GloVe Algorithm

- **GloVe**: Stands for **Global Vectors for Word Representation**.

- Competes with **Word2Vec** (Skip-gram and CBOW) for word embedding generation.
- Known for its **simplicity and interpretability** in word vector learning.
---
- GloVe builds a global word-word co-occurrence matrix.
- Learns embeddings where the dot product approximates log co-occurrence.
- Simple yet effective; retains analogy relationships.
- Embeddings are not easily interpretable in individual dimensions, but still useful for downstream tasks.
---


## Co-occurrence Matrix
- Let **X_ij** be the number of times word *i* appears in the context of word *j*.
- **Symmetry**: If context is defined as ±10 words, then **X_ij = X_ji**.
- Asymmetric cases exist (e.g., using only the previous word as context).

##  Objective Function
- Minimize:  
  `J = Σ_i Σ_j f(X_ij) * (θᵢᵀ * eⱼ - log(X_ij))²`
- θᵢ: vector representation of word *i*.  
  eⱼ: vector representation of context word *j*.
- The model learns **word vectors** such that their dot product predicts the **log of co-occurrence count**.

---
- Any invertible matrix A can transform embeddings without changing objective:
  - `θᵢᵀ * eⱼ ≡ (Aθᵢ)ᵀ * (A⁻¹ᵀ eⱼ)`
- Confirms that the embeddings may not be directly interpretable per-dimension.
---
- Despite the simple squared-error formulation, the algorithm is powerful.
- Emerged from attempts to simplify previous models like the **neural language model** and **Word2Vec**.
---
# Sentiment Classification with Word Embeddings

- **Challenge in Sentiment Classification**
  - Often have **limited labeled training data** (sometimes <10,000 examples).
  - Despite this, **word embeddings** help build effective classifiers with small data

## Basic Sentiment Classification Model

- **Pipeline**
  1. Convert each word in the sentence to a one-hot vector.
  2. Multiply by embedding matrix `E` to get embedding vectors.
  3. **Sum or average** the embeddings.
  4. Feed the resulting vector into a **softmax classifier** to predict sentiment class.

- **Benefits**
  - Works with inputs of **variable lengths**.
  - Simple to implement.
  - Leverages pretrained word knowledge.

- **Limitations**
  - **Ignores word order**.
  - Can misclassify sentences like:
    > “Completely lacking in good taste, good service, and good ambiance.”
    - Words like “good” dominate due to averaging, leading to incorrect classification.

---

## Improved Model: RNN-Based Sentiment Classification

- **Pipeline**
  1. Convert sentence words to one-hot vectors.
  2. Multiply by embedding matrix `E` to get word embeddings.
  3. Feed embeddings into an **RNN** (Recurrent Neural Network).
  4. Use the **last hidden state** as input to predict the sentiment.

- **Advantages**
  - **Captures word sequence** and dependencies.
  - Learns that phrases like “not good” are negative, unlike bag-of-words models.
  - Handles **complex structures and negations**.

- **Generalization Capability**
  - Can understand new/unseen words like “absent” if present in the embedding training corpus.
  - Helps overcome lack of label data by leveraging **large unsupervised corpora**.

---
# Reducing Bias in Word Embeddings

- **Bias reduction in word embeddings is essential** as AI systems are used in decision-making.
- This method includes:
  1. Identifying bias direction using SVD
  2. Neutralizing non-definitional words
  3. Equalizing definitional word pairs
- **Still an active research area** with ongoing advancements.
- These techniques help promote fairer AI systems and more ethical use of machine learning.
---
# Sequence-to-Sequence Models & Image Captioning

- **Sequence-to-sequence (seq2seq) models** are used for tasks like machine translation and speech recognition.
- Introduced in two influential papers:

## Sequence-to-Sequence Model for Machine Translation

### Task Example
- **Input**: French sentence: *Jane visite l'Afrique Septembre*
- **Output**: English translation: *Jane is visiting Africa in September*

### Model Structure
- **Encoder**:
  - Implemented as an RNN (GRU or LSTM)
  - Reads the input sequence word-by-word
  - Outputs a fixed-length **context vector** representing the entire input sentence
- **Decoder**:
  - Takes the context vector as input
  - Outputs translated words one at a time
  - Stops when it emits an end-of-sequence token

### Training
- Train using many pairs of input (e.g., French) and output (e.g., English) sequences
- Use teacher forcing during training (feed ground-truth word as next input)

### Remarkable Result
- With sufficient data, this basic model architecture performs **surprisingly well** for machine translation tasks.

---

## Image Captioning with Sequence Models

### Task Example
- **Input**: Image (e.g., cat sitting on a chair)
- **Output**: Text caption: *A cat sitting on a chair*

### Model Architecture
- **Image Encoder**:
  - Use a pre-trained CNN (e.g., AlexNet) without the final softmax layer
  - Outputs a high-dimensional feature vector (e.g., 4096D) representing the image
- **Text Decoder**:
  - An RNN takes the image feature vector as input
  - Outputs the caption one word at a time

---
- Image-to-sequence models operate similarly to sequence-to-sequence models in NLP
- Especially effective when captions are **short and simple**
- Multiple groups proposed similar image captioning models independently around the same time:
  - Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Alan L. Yuille
  - Andrej Karpathy and Fei-Fei Li

---

## Key Differences from Language Modeling
- **Language modeling** may sample randomly from the next-word distribution
- **Seq2seq translation/captioning**: Often aim for the **most likely** sequence rather than sampling randomly
  - Important to generate coherent and accurate outputs

---

# Conditional Language Models & Machine Translation

## Language Model (Baseline)
- Trained to estimate the **probability of a sentence**.
- Can be used to generate novel sentences by sampling one word at a time.
- Often starts generation with a vector of all zeros.
- Words are generated sequentially: output word at time `t-1` is fed as input at time `t`.

---

## Machine Translation Model

### Architecture
- Composed of:
  - **Encoder network** (green): Processes input French sentence into a context vector.
  - **Decoder network** (purple): Generates English translation one word at a time.
- Decoder is similar to a standard language model, but starts with the encoded input vector instead of zeros.

### Conditional Language Modeling
- The model estimates the probability of an English translation **conditioned on** the French input:
  ```
  p(y | x)
  ```
  Where `x` is the French sentence and `y` is the English translation.

---

## Translation as Conditional Probability
- Given a French input, the model predicts the probabilities of various English outputs.
- **Example**:
  - French input: *Jane visite l’Afrique en septembre*
  - Model computes the probability of several English sentences like:
    - *Jane is visiting Africa in September*
    - *In September, Jane will visit Africa*
    - *Her African friend welcomed Jane in September*
- These outputs vary in quality — some are good, some are verbose, and some incorrect.

---

## Why Not Random Sampling?
- Sampling randomly from `p(y | x)` might give:
  - Correct but verbose translations
  - Awkward phrasing
  - Incorrect translations
- **Goal**: Find the **most likely** translation, not just any plausible one.

---

## Greedy Search (Naive Approach)
- Greedy search generates the sentence by:
  1. Picking the most likely word at each step.
  2. Choosing the best next word **based only on the current sequence**.
- **Problem**: This can lead to suboptimal translations.
  - Example: Preferring “Jane is going…” over “Jane is visiting…” because “going” is more common.
  - Greedy decisions early on can trap the model into poor completions.

---

## Search Space Challenge
- The number of possible English sentences is **exponentially large**.
  - For 10-word outputs with a 10,000-word vocabulary:
    - \( 10,000^{10} \) possible combinations
- Exhaustive search is computationally infeasible.

---

## Solution: Approximate Search Algorithms
- Use search algorithms to find high-probability sentences without evaluating all combinations.
- Goal: Find a good approximation of the sentence that **maximizes** `p(y | x)`.
- Common method: **Beam Search** (explained in the next lesson)
  - Efficient and typically yields high-quality translations.
  - Not guaranteed to find the absolute best sentence, but works well in practice.

---

# Beam Search Algorithm 



- **Beam Search** is a widely used algorithm to find the most likely output sentence from a model.
- It is an **approximate search algorithm** used to select the best output sequence by exploring multiple possibilities at each step.

---

## How Beam Search Works

### Step 1: Choose the First Word
- Given the input (e.g., French sentence), use the encoder-decoder model to evaluate the probability of each word in the vocabulary as the **first word**.
- **Beam Width (B)**: Number of options to consider at each step.  
  - Example: If B = 3, keep the top 3 most likely first words (e.g., "in", "jane", "september").

### Step 2: Expand to Second Word
- For each of the top B first words:
  - Feed the decoder with that word and compute the probabilities for all possible second words (10,000 vocabulary size).
  - This results in B × 10,000 combinations (e.g., 30,000 if B=3).
- **Score = P(y1|x) × P(y2|y1, x)** → Select top B combinations.

### Step 3: Third Word and Beyond
- For each of the selected B pairs (y1, y2), repeat the process:
  - Use the decoder, feed in y1 and y2, compute probabilities for y3.
  - Score each triplet: **P(y1|x) × P(y2|y1,x) × P(y3|y1,y2,x)**
  - Again, select top B combinations and continue the process.
- Repeat until the `<EOS>` (end of sentence) token is generated.

---

# Beam Search - Refinements

## Length Normalization

- **Problem with raw probabilities**: Multiplying many probabilities (all < 1) leads to very small numbers → *numerical underflow*.
- **Solution – Use log probabilities**:  
  - Convert product of probabilities into sum of log-probabilities for numerical stability.  
  - `log(p(y | x)) = sum(log(p(yt | x, y1,...,yt-1)))`
  - Logarithm is strictly increasing → maximizing log-probability still gives same best sequence as original.

---

## Penalizing Long Sentences (and Fix)

- **Issue with long outputs**:
  - Multiplying many <1 values → smaller probability.
  - Log-probability sum becomes more negative → discourages long translations.
- **Fix – Normalize by sentence length**:
  - Instead of summing log-probabilities, use average:
    - `1/Ty * sum(log(p(yt | ...)))`
  - Reduces bias against longer outputs.
  - Prevents over-preference for short translations.

---

## Heuristic – Length Normalization Power (α)

- **Generalized normalization**:  
  - Use `sum(log(p)) / Ty^α` instead of `Ty` directly.
  - Hyperparameter `α` ∈ [0, 1]:
    - α = 0 → no normalization (raw sum)
    - α = 1 → full normalization (average)
    - Typical value: `α ≈ 0.7`


---

## Search Steps and Final Output

- **Track top sentences at each length**:  
  - Run beam search for `N` steps (e.g., N = 30).
  - Keep beam width `B` hypotheses at each step.
- **Final scoring**:  
  - Re-score all seen sentence candidates using normalized log-probability.
  - Output sentence with **highest normalized score**.

---

## Choosing Beam Width (B)

- **Large beam width**:
  - Pros: More search paths → better result.
  - Cons: Slower, more memory intensive.
- **Small beam width**:
  - Pros: Fast and memory efficient.
  - Cons: Worse performance due to limited hypothesis.

- **Diminishing returns**:
  - Big gains going from 1 → 3 → 10.
  - Less improvement beyond 100s or 1000s.

---

## Beam Search vs Classic Search Algorithms

- **Not exact search**:
  - Beam Search ≠ BFS/DFS from traditional CS.
  - It is an *approximate* search algorithm.
  - Faster, but not guaranteed to find global optimum (true argmax).
- **Trade-off**:
  - Accuracy vs computation.

---

# Error Analysis with Beam Search

## Purpose of Error Analysis
- Helps identify whether performance issues are due to the **beam search algorithm** or the **RNN model**.
- Enables more efficient use of time by focusing on the true bottleneck (model vs search).

## Components of the System
- **RNN Model (Encoder-Decoder)**: Computes \( P(y|x) \), i.e., probability of output translation given input.
- **Beam Search Algorithm**: Approximate (heuristic) method to find the translation \( y \) that maximizes \( P(y|x) \).

## Common Mistake Example
- **Human Translation** (y\*): "Jane visits Africa in September"
- **Beam Search Output** (ŷ): "Jane visited Africa last September"
- The output ŷ changes the meaning — it’s an incorrect translation.

## Outcome of Analysis
- **If most errors are due to Beam Search**:
  - Consider increasing beam width.
- **If most errors are due to RNN**:
  - Consider model improvements: more data, regularization, different architecture, etc.
---

# BLEU Score 


- **BLEU Score**: Stands for *Bilingual Evaluation Understudy*. It is a metric for automatically evaluating the quality of machine-generated translations against human references.

## Motivation
- **Multiple Valid Translations**: Unlike tasks like image classification, machine translation can have many correct outputs for a given input.
- **Need for Automation**: BLEU acts as an “understudy” to human evaluation, providing a quick and automatic scoring method.
- **Use Cases**: Commonly used in machine translation, image captioning, and other text generation tasks where multiple outputs may be valid.

## Basic Idea
- **Goal**: Measure how close the machine translation (MT) output is to one or more human references.
- **Approach**: Compute overlap of *n-grams* between MT output and reference translations.

## Modified Precision
- **Precision**: Initially measured by how many words in the MT output appear in references.
- **Issue**: A bad sentence like “the the the the the” could have high precision if words repeat.
- **Modified Precision**: 
  - **Clip counts** to the maximum number of times a word appears in any reference.
  - **Example**: If "the" appears max 2 times in references, only 2 of 7 "the"s are counted → Precision = 2/7.

## N-gram Precision
- **N-grams**: Consecutive word sequences (e.g., unigrams, bigrams, trigrams).
- **Bigram Example**: 
  - MT output: "the cat the cat on the mat"
  - Bigrams: "the cat", "cat the", "cat on", "on the", "the mat"
  - **Clipping**: Only count each bigram up to how often it appears in references.

## Modified Precision Formula
- For unigrams:
  ```
  P₁ = Σ (clipped count of unigram) / Σ (total count of unigram in MT output)
  ```
- For n-grams:
  ```
  Pₙ = Σ (clipped count of n-gram) / Σ (total count of n-gram in MT output)
  ```

## Final BLEU Score
- **Combine Precisions**: Compute average of P₁, P₂, P₃, P₄
  ```
  BLEU = BP * exp(Σₙ wₙ * log(Pₙ))  (usually wₙ = 1/4 for n=1 to 4)
  ```
- **BP (Brevity Penalty)**: 
  - Penalizes short translations (as short outputs tend to inflate precision).
  - Formula:
    - If MT output length > reference length: BP = 1
    - Else: BP = exp(1 - r/c)  
      - where `r` = length of reference, `c` = length of MT output

## Key Properties
- **Perfect Score**: BLEU = 1 if MT output matches a reference exactly.
- **Interpretable Metric**: Helps compare systems quickly via a single number.
- **Caveats**: Not perfect. Doesn’t always capture meaning or fluency.
---

# Attention Model 

## Introduction to the Attention Model
- **Encoder-Decoder Architecture Limitations**: Traditional encoder-decoder RNNs read the entire input sentence into a fixed-size vector before decoding, which works well for short sentences but struggles with long sentences.
- **Motivation for Attention**: Long sentences reduce performance (e.g., BLEU score drops) because memorizing the entire input in a single vector is hard. Human translators translate incrementally, focusing on one part of the input at a time.

## Performance Trends
- **BLEU Score Analysis**:
  - High BLEU for short sentences.
  - Significant drop for long sentences (30–40+ words).
- **Attention Fixes This**: Allows the model to “look at” relevant parts of the input at each decoding step, improving translation quality even for long sentences.

## Key Idea of Attention
- **Incremental Focus**: The model learns to pay attention to specific words in the input while generating each word in the output, similar to how humans work.
- **Bidirectional RNN Encoder**:
  - Processes input sentence and creates feature vectors for each input position.
  - Captures context from both directions.

## Decoder with Attention
- **Hidden States**: Uses `Sₜ` to denote decoder hidden states to avoid confusion with encoder activations.
- **Context Vector `Cₜ`**:
  - Dynamically computed at each time step.
  - Aggregates relevant encoder features weighted by attention scores.

## Attention Weights (α)
- **Attention Score αₜₜ'**: When generating the t-th output word, αₜₜ' represents the attention weight on the t'-th input word.
- **Weights are Learned**:
  - At each decoder step `t`, a set of αₜₜ' values are computed over all encoder time steps.
  - Used to compute the context vector `Cₜ` via a weighted sum of encoder activations.

## Decoding Steps with Attention
- **Step-by-step Generation**:
  - Step 1: Generate first output word (`Jane`), compute attention weights α₁₁ to α₁₅ over French input.
  - Step 2: Generate second word (`visits`), use `S₁`, previous word, and new α₂₁ to α₂₅ for updated context.
  - Step 3: Continue similarly, each time using prior state, context vector, and attention weights.
- **EOS Generation**: Continues until end-of-sequence token is generated.
---



# Attention Model


## Input Representation
- A **bidirectional RNN/GRU/LSTM** is used to process the input sentence.
- Each word in the input is encoded into features:
  - `→ a_forward(t')` - Forward activation.
  - `← a_backward(t')` - Backward activation.
- These are concatenated to form `a(t')`, the annotation vector (feature vector) for the input word at time step `t'`.

## Output Generation
- A **unidirectional RNN** generates the output translation one word at a time.
- At each time step `t`, the RNN uses:
  - The previous hidden state `s(t-1)`.
  - A context vector `c(t)`, computed using attention.

## Attention Weights
- Define `α(t, t')` as the **attention weight**: how much output word `y(t)` should attend to input annotation `a(t')`.
- Attention weights are:
  - Non-negative.
  - Sum to 1 (for each output time step `t`).

## Context Vector Computation
- At time step `t`, the context vector `c(t)` is computed as:
  ```
  c(t) = Σ over t' of α(t, t') * a(t')
  ```
- Weighted sum of all input annotations `a(t')`, using attention weights `α(t, t')`.

## Computing Attention Weights
- Use a small neural network to compute intermediate scores `e(t, t')`:
  - Inputs:
    - `s(t-1)` - Hidden state from previous decoder time step.
    - `a(t')` - Input annotation at position `t'`.
  - Output: `e(t, t')` - Raw attention score.
- Apply **softmax** over `e(t, t')` values for fixed `t` to get normalized attention weights:
  ```
  α(t, t') = softmax(e(t, t'))
  ```

## Why This Works
- `s(t-1)` contains information about what the model has generated so far.
- `a(t')` contains information about each input word.
- Neural network learns how to combine these using backpropagation.

##  Training
- Entire model (encoder + attention + decoder) is trained end-to-end using gradient descent.
- The attention mechanism automatically learns **where to focus** for each word being generated.

## Computational Cost
- Time complexity is **O(Tx * Ty)** where:
  - `Tx`: length of input sequence.
  - `Ty`: length of output sequence.
- Quadratic cost is usually acceptable in practice for translation tasks with short to medium sequences.

## Applications Beyond Translation
- Attention is also used in:
  - **Image captioning**: Focus on different regions of an image while generating each word.
    - See work by Kevin Chu et al. (Bengio et al.)
  - **Date normalization task** :
    - Input: various date formats (e.g., `"April 20 1969"`).
    - Output: normalized date (e.g., `"1969-04-20"`).
    - Model uses attention to focus on parts of the date string.

## 🔍 Visualizing Attention
- Attention weights `α(t, t')` can be visualized as a heatmap:
  - Rows: Output words.
  - Columns: Input words.
  - Brightness indicates weight magnitude.
- Helps interpret model behavior—shows it often aligns with intuitive word correspondences.




