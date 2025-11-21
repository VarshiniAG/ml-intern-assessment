import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention (batch-compatible).
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    mask: (batch, seq_q, seq_k) optional
    Returns: output, attention_weights
    """
    # 1. Compute raw attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # batch matmul

    # 2. Scale
    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)

    # 3. Apply mask
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 4. Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. Weighted sum of values
    output = np.matmul(attention_weights, V)

    return output, attention_weights
