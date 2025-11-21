# AIML Intern Assignment – Project Evaluation

## 1. Overview
This project implements **Scaled Dot-Product Attention** and a **Trigram Language Model** as part of the AIML internship assignment. The focus is on **modular design, reproducibility, testability, and compatibility with Google Colab**.  

The goal was to understand and implement **core sequence modeling techniques** used in modern NLP architectures, such as transformers.

---

## 2. Design Choices

### a. Scaled Dot-Product Attention
- **Scaling by √dₖ:** Ensures numerical stability and prevents softmax from saturating.  
- **Masking Support:**
  - **Causal mask:** Blocks attention to future tokens for autoregressive tasks.  
  - **Padding mask:** Ignores padding tokens in variable-length sequences.  
- **Batch processing:** Handles both single sequences and batched inputs efficiently.  
- **Vectorized operations:** Used **NumPy matrix multiplications** for computational efficiency.

### b. Trigram Language Model
- **Probabilistic text generation:** Uses a bigram context to predict the next word.  
- **Tokenization & Preprocessing:** Handles punctuation, lowercasing, and unknown tokens.  
- **Reproducibility:** Random seed setting ensures consistent outputs for experiments.  

### c. Modular Utilities
- Functions for **mask creation, text cleaning, tokenization**, and **seed management**.  
- Designed for **reusability** across experiments and future model extensions.  

---

## 3. Testing & Validation
- **Unit tests** implemented for all core components (`attention.py` and `utils.py`).  
- Tested on both **single** and **batched sequences** for attention.  
- Verified **masking correctness** in attention computations.  
- All tests passed successfully in **Google Colab** environment, ensuring compatibility.  

---

## 4. Learning Outcomes
1. **Deep understanding of attention mechanisms:** Implemented scaled dot-product attention from scratch.  
2. **Probabilistic sequence modeling:** Built a trigram-based text generator with correct handling of edge cases.  
3. **Software engineering practices:** Modular code design, unit testing, reproducibility, and clear documentation.  
4. **Numerical stability considerations:** Learned the importance of scaling and softmax normalization.  
5. **Hands-on experience with NumPy:** Applied matrix operations for efficient ML computations.  

---

## 5. Challenges & Resolutions
| Challenge | Resolution |
|-----------|-----------|
| Handling batched inputs for attention | Added batch dimension handling and vectorized operations |
| Ensuring numerical stability | Scaled dot-product by √dₖ before applying softmax |
| Masking for causal/padding scenarios | Implemented flexible mask creation in `utils.py` |
| Reproducibility in text generation | Added random seed setting for deterministic outputs |
| Alignment errors in matrix multiplications | Corrected dot-product shapes for batch and sequence dimensions |

---

## 6. Project Impact
- Demonstrates **core understanding of transformer components**.  
- Provides **reusable codebase** for future NLP or sequence modeling projects.  
- Fully compatible with **Google Colab**, making it easy to experiment, debug, and showcase results.  
- Prepares the intern for **advanced AI/ML tasks** in production-level environments.  

---

## 7. Future Improvements
- Extend attention to **multi-head attention**.  
- Implement **positional encodings** for full transformer simulations.  
- Integrate with larger NLP pipelines for **text classification or generation** tasks.  
- Optimize for **GPU acceleration** for larger datasets.

---

**Prepared by:** Varshini A G  
**GitHub:** [github.com/VarshiniAG](https://github.com/VarshiniAG)


