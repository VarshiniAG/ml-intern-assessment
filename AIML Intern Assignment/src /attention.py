# src/attention.py

import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries matrix of shape (seq_len_q, d_k)
        K: Keys matrix of shape (seq_len_k, d_k)
        V: Values matrix of shape (seq_len_v, d_v)
        mask: Optional mask matrix of shape (seq_len_q, seq_len_k). Positions with 0 are masked.

    Returns:
        output: Attention-applied output of shape (seq_len_q, d_v)
        attention_weights: Softmax-normalized attention weights (seq_len_q, seq_len_k)
    """
    # Step 1: Compute raw attention scores by Q @ K^T
    scores = np.dot(Q, K.T)

    # Step 2: Scale by sqrt(d_k) to prevent large values
    d_k = K.shape[1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (if provided) by setting masked positions to -inf
    if mask is not None:
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)

    # Step 4: Apply softmax to get attention weights
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: Compute final output by multiplying weights with values
    output = np.dot(attention_weights, V)

    return output, attention_weights

