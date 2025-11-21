# tests/test_attention.py

import numpy as np
from src.attention import scaled_dot_product_attention
from src.utils import generate_sample_matrices, generate_mask

def test_scaled_dot_product_attention():
    # Generate sample Q, K, V
    Q, K, V = generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2)

    # Optional mask (masking position (0,2) for demonstration)
    mask = generate_mask(seq_len_q=3, seq_len_k=3, mask_indices=[(0, 2)])

    # Run attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

    # Print results
    print("Q:\n", Q)
    print("K:\n", K)
    print("V:\n", V)
    print("Mask:\n", mask)
    print("Attention Weights:\n", attn_weights)
    print("Output:\n", output)

    # Basic assertions
    assert attn_weights.shape == (Q.shape[0], K.shape[0]), "Attention weights shape mismatch"
    assert output.shape == (Q.shape[0], V.shape[1]), "Output shape mismatch"
    assert np.allclose(np.sum(attn_weights, axis=1), 1), "Attention weights not normalized"

if __name__ == "__main__":
    test_scaled_dot_product_attention()

