# AIML Intern Assignment – Scaled Dot-Product Attention & Trigram Language Model

**Empowering models to focus, understand, and generate sequences with context-awareness.**  
This project implements **Scaled Dot-Product Attention** and a **Trigram Language Model** from scratch as part of the AIML internship assignment, with emphasis on **modularity, readability, and test-driven development**. Fully compatible with **Google Colab** for seamless experimentation.


# Key Highlights
- Attention Mechanism from Scratch:** Understand and implement the core of transformer architectures.  
- Trigram Language Model:** Probabilistic text generation based on bigram context.  
- Test-Driven Development:** All functions validated with **unit tests** to ensure correctness.  
- Reusable Utilities:** Includes masking, tokenization, and reproducible seed settings.  
- Colab-Ready:** Run experiments and visualize outputs interactively.


# Scaled Dot-Product Attention

# Concept
Attention lets a model **focus on relevant input features** when producing outputs.  
**Scaled Dot-Product Attention** computes attention scores using Queries (Q) and Keys (K), scales them for stability, and applies them to Values (V).

# Formula
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\]

- *Q:* Query matrix  
- *K:* Key matrix  
- *V:* Value matrix  
- *dₖ:* Key dimension (for numerical stability)  

# Features
- **Scaling by √dₖ:** Prevents large values in softmax, ensures gradient stability.  
- **Masking:** Supports **causal masks** for autoregressive tasks and **padding masks** for variable-length sequences.  
- **Batch Support:** Handles single and batched inputs seamlessly.  

---

# Project Structure

      AIML_Intern_Assignment/
           ├── src/
           │ ├── attention.py # Scaled Dot-Product Attention
           │ ├── utils.py # Utilities: masking, tokenization, seed handling
           │ └── init.py
           ├── tests/
           │ └── test_attention.py # Unit tests for attention
           ├── notebooks/ # Optional experimentation notebooks
           ├── README.md
           └── evaluation.md # Insights, design choices, and performance



# Quick Start

# Clone & Navigate

     !git clone https://github.com/Priya-96-aiml/ml-intern-assessment.git
     %cd "ml-intern-assignment/AIML Intern assignment"
Install Dependencies

    !pip install numpy pytest
Run Attention Example

    import numpy as np
    from src.attention import scaled_dot_product_attention
    from src.utils import create_mask

# Sample matrices
    Q = np.array([[[1, 2], [3, 4]]])
    K = np.array([[[1, 2], [3, 4]]])
    V = np.array([[[1, 2], [3, 4]]])

# Compute attention
    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Attention Output:\n", output)
    print("Attention Weights:\n", weights)

# With causal mask
    mask = create_mask(Q.shape[1], mode="causal")
    out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Attention Output with Causal Mask:\n", out_masked)
Run Unit Tests

    !pytest -q
# - All tests should pass.

# Design Insights
- Causal & Non-Causal Masks: Correct attention in sequential tasks.

- Numerical Stability: Scaling ensures softmax outputs are well-behaved.

- Modular Utilities: Reusable functions for tokenization, masking, and random seed handling.

- Trigram Model: Demonstrates probabilistic sequence generation with context-awareness.

# Learning Outcomes
- Implemented a core transformer component from scratch using Python & NumPy.

- Built robust, testable, and modular code for professional-quality ML projects.

- Learned to handle edge cases like empty inputs and unseen n-grams.

- Fully Colab-compatible for experimentation and demonstrations.
