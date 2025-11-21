# tests/test_attention.py

import sys
import os
import numpy as np

# Fix path so tests can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from attention import scaled_dot_product_attention
from utils import generate_sample_matrices, generate_mask

# ------------------------------
# Test 1: Check output and weight shapes
# ------------------------------
def test_attention_shapes():
    Q, K, V = generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2)
    # Add batch dimension
    output, weights = scaled_dot_product_attention(Q[None,:,:], K[None,:,:], V[None,:,:])
    assert output.shape == (1, 3, 2), "Output shape mismatch"
    assert weights.shape == (1, 3, 3), "Attention weights shape mismatch"

# ------------------------------
# Test 2: Check softmax sum to 1
# ------------------------------
def test_attention_softmax_sum():
    Q, K, V = generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2)
    output, weights = scaled_dot_product_attention(Q[None,:,:], K[None,:,:], V[None,:,:])
    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, rtol=1e-5)

# ------------------------------
# Test 3: Masking works correctly
# ------------------------------
def test_attention_with_mask():
    Q, K, V = generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2)
    mask = generate_mask(seq_len_q=3, seq_len_k=3, mask_indices=[(0,2)])  # mask one element
    output, weights = scaled_dot_product_attention(Q[None,:,:], K[None,:,:], V[None,:,:], mask=mask)
    # Masked position should have near-zero attention
    assert weights[0, 0, 2] < 1e-5, "Mask not applied correctly"

# ------------------------------
# Test 4: Batch attention
# ------------------------------
def test_attention_batch():
    batch_size = 2
    Q, K, V = generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2)
    Q_batch = np.stack([Q, Q])
    K_batch = np.stack([K, K])
    V_batch = np.stack([V, V])
    output, weights = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
    assert output.shape == (batch_size, 3, 2), "Batch output shape mismatch"
    assert weights.shape == (batch_size, 3, 3), "Batch weights shape mismatch"
    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, rtol=1e-5)

# ------------------------------
# Run tests if executed directly
# ------------------------------
if __name__ == "__main__":
    import pytest
    pytest.main(["-v", "--disable-warnings"])
