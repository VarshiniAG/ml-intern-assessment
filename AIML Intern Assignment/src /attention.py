# src/attention.py

import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention with batch support.

    Args:
        Q: Queries of shape (batch_size, seq_len_q, d_k)
        K: Keys of shape (batch_size, seq_len_k, d_k)
        V: Values of shape (batch_size, seq_len_v, d_v)
        mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)
              or (seq_len_q, seq_len_k). Positions with 0 are masked.

    Returns:
        output: Attention-applied output of shape (batch_size, seq_len_q, d_v)
        attention_weights: Softmax-normalized weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    # Step 1: Compute attention scores using batch matrix multiplication
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1)))  # (batch, seq_q, seq_k)

    # Step 2: Scale scores by sqrt(d_k)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask if provided
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, :, :]  # broadcast for batch
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)

    # Step 4: Softmax over last dimension (keys)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: Multiply attention weights by values
    output = np.matmul(attention_weights, V)  # (batch, seq_q, d_v)

    return output, attention_weights


# Optional demo run
if __name__ == "__main__":
    Q = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    K = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    V = np.array([[[1, 2], [3, 4]]], dtype=np.float32)

    out, weights = scaled_dot_product_attention(Q, K, V)
    print("Output:\n", out)
    print("Weights:\n", weights)
