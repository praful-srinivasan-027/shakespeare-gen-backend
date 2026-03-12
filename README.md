# Speak, Shakespeare | Backend API & Inference Engine

This repository contains the backend infrastructure, neural network architecture, and pre-trained weights for the **Speak, Shakespeare** generative text engine. The service is built using **FastAPI** and **PyTorch**, serving a custom-trained **Gated Recurrent Unit (GRU)** model that specializes in character-level linguistic modeling.

---

##  Deep Learning Architecture: Character-Level GRU

The backend operates on a character-level sequence-to-sequence logic. Unlike traditional Word-RNNs, this model predicts the next character based on the preceding sequence of characters, allowing it to learn not just words, but the unique punctuation, archaic syntax, and rhythmic structure characteristic of 16th-century prose.

### 1. Neural Network Topology (`model.py`)
The model is defined by the `CharRNN` class, which implements a sophisticated pipeline to transform raw text into probabilistic predictions:

* **Embedding Layer:** A `nn.Embedding` layer that maps each character ID to a 10-dimensional dense vector. This allows the model to learn semantic relationships between characters (e.g., that 'q' is almost always followed by 'u').
* **Recurrent Layer (GRU):** A 2-layer Gated Recurrent Unit. GRUs are utilized here over standard RNNs because they effectively solve the "vanishing gradient" problem through update and reset gates, allowing the model to remember context from the beginning of a sentence.
    * **Hidden Dimension:** 100 units per layer.
    * **Dropout:** 0.2, applied to the hidden states between GRU layers to enforce feature redundancy and prevent overfitting.
* **Linear Output:** A fully connected layer that projects the 100-dimensional hidden state back into the 39-dimensional vocabulary space, producing "logits" for each possible next character.

### 2. Training Methodology (`char_RNN.ipynb`)
The model was trained on the **Complete Works of William Shakespeare**, using a sliding window approach:
* **Sequence Length:** 100 characters. For every step, the model looks at 100 characters to predict the 101st.
* **Optimization:** The model was trained using **Cross-Entropy Loss** and the **Adam Optimizer**, reaching roughly **1800 epochs**.
* **Validation:** It achieved a **Test Accuracy of 43.43%**. In character prediction (where there are 39 possible choices), an accuracy of ~43% indicates the model has successfully mastered common English patterns and specific Shakespearean vocabulary.

---

##  Inference & Sampling Logic

The generation process is not strictly deterministic; it uses **Multinomial Sampling** to ensure the output feels "alive" and varied.

### The Role of Temperature
The `get_char` function applies a **Temperature ($T$)** scaling factor to the model's output logits before passing them through a Softmax function:
$$P(x_i) = \frac{e^{logits_i / T}}{\sum e^{logits_j / T}}$$

* **Low Temperature (e.g., 0.2):** Makes the probability distribution "sharp." The model will almost always pick the most likely character, leading to very coherent but often repetitive text.
* **High Temperature (e.g., 0.9):** Flattens the distribution. The model is more likely to pick "surprising" characters, leading to high creativity but increasing the risk of spelling errors or "gibberish."

---

##  API Specification

The API is served via **FastAPI** with optimized endpoints for real-time inference.

### `GET /generate`
**Request Parameters:**
* `text` (str): The seed string (e.g., "To be or not").
* `num_chars` (int): Number of characters to generate (e.g., 200).
* `temperature` (float): Creativity threshold (0.1 to 1.0).

**Internal Process:**
1.  **Normalization:** Input text is converted to lowercase.
2.  **Encoding:** Text is converted to a tensor of integer IDs using the `char_to_id` map.
3.  **Looping Inference:** The model generates one character at a time. Each new character is appended to the input sequence to inform the prediction of the next character.
4.  **Decoding:** The resulting integer IDs are mapped back to characters using the `id_to_char` map.

---

##  Deployment & Environment

* **Framework:** FastAPI
* **Production Server:** Uvicorn
* **Hosting:** Render (Web Service)
* **Python Version:** 3.11 (configured via `runtime.txt`)
* **CORS Policy:** Enabled for all origins (`*`) to facilitate cross-origin requests from the React frontend.

### Dependency Management
The environment is kept lightweight to minimize "cold start" times on Render:
```text
fastapi
uvicorn
torch
pydantic
