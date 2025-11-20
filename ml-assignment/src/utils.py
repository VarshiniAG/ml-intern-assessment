import re
import logging
import random
from typing import List

def clean_text(text: str) -> str:
    """
    Cleans text by:
    - converting to lowercase
    - removing non-alphanumeric characters
    - collapsing multiple spaces
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Keep only alphabets, digits, apostrophes, and spaces
    cleaned = re.sub(r"[^a-z0-9'\s]+", " ", text)

    # Normalize multiple spaces
    cleaned = " ".join(cleaned.split())

    return cleaned


def tokenize(text: str) -> List[str]:
    """
    Splits cleaned text into tokens.
    """
    if not text:
        return []
    return text.split()


def set_seed(seed: int = 42):
    """
    Sets seeds for Python, NumPy (if available), and random for reproducibility.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # NumPy not installed, skip.


def get_logger(name: str = "app_logger"):
    """
    Returns a simple logger instance with formatting.
    Prevents adding duplicate handlers on repeated imports.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger# This file is optional.
# You can add any utility functions you need for your implementation here.
