# src/utils.py

import numpy as np

def generate_sample_matrices(seq_len_q=3, seq_len_k=3, d_k=3, d_v=2, seed=42):
    """
    Generate sample Q, K, V matrices for testing attention.

    Args:
        seq_len_q: Number of queries
        seq_len_k: Number of keys/values
        d_k: Dimension of queries/keys
        d_v: Dimension of values
        seed: Random seed for reproducibility

    Returns:
        Q, K, V: numpy arrays
    """
    np.random.seed(seed)
    Q = np.random.randint(0, 10, size=(seq_len_q, d_k)).astype(float)
    K = np.random.randint(0, 10, size=(seq_len_k, d_k)).astype(float)
    V = np.random.randint(0, 10, size=(seq_len_k, d_v)).astype(float)
    return Q, K, V

def generate_mask(seq_len_q, seq_len_k, mask_indices=None):
    """
    Generate a mask matrix with 0s at masked positions and 1s elsewhere.

    Args:
        seq_len_q: Number of queries
        seq_len_k: Number of keys
        mask_indices: List of (i,j) positions to mask

    Returns:
        mask: numpy array of shape (seq_len_q, seq_len_k)
    """
    mask = np.ones((seq_len_q, seq_len_k))
    if mask_indices:
        for i, j in mask_indices:
            mask[i, j] = 0
    return mask

