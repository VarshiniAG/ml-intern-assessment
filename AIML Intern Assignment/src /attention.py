## src/attention.py

import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries matrix of shape (batch_size, seq_len_q, d_k)
        K: Keys matrix of shape (batch_size, seq_len_k, d_k)
        V: Values matrix of shape (batch_size, seq_len_v, d_v)
        mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k) 
              or (seq_len_q, seq_len_k). Positions with 0 are masked.

    Returns:
        output: Attention-applied output of shape (batch_size, seq_len_q, d_v)
        attention_weights: Softmax-normalized attention weights 
                           of shape (batch_size, seq_len_q, seq_len_k)
    """

    # Step 1: Compute raw attention scores (batch matmul)
    # scores shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1)))

    # Step 2: Scale scores by sqrt(d_k)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (if provided)
    if mask is not None:
        # Broadcast mask if necessary
        if mask.ndim == 2:
            mask = mask[None, :, :]
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)

    # Step 4: Softmax to get attention weights
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: Multiply attention weights with values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


# Optional: Demo run if executed directly
if __name__ == "__main__":
    # Example batch with 1 sequence of length 2 and dimension 2
    Q = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    K = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    V = np.array([[[1, 2], [3, 4]]], dtype=np.float32)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Attention Output:\n", output)
    print("Attention Weights:\n", weights)
