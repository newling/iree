"""
Compute expected output for reduction and add

Usage:
    python create_expected.py arg0.npy arg1.npy

Output:
    expected0.npy in the current directory
"""

import sys
import numpy as np
from pathlib import Path


def load_array(path: str) -> np.ndarray:
    """Load a numpy array from a .npy file."""
    return np.load(path)


def save_array(path: str, array: np.ndarray):
    """Save a numpy array to a .npy file."""
    np.save(path, array)


def compute_expected(arg0: np.ndarray, arg1: np.ndarray) -> np.ndarray:
    reduced = np.sum(arg0, axis=(0,))
    return reduced + arg1


def main(argv):
    if len(argv) != 3:
        print("Usage: python create_expected.py arg0.npy arg1.npy")
        sys.exit(1)

    arg0 = load_array(argv[1])
    arg1 = load_array(argv[2])

    try:
        result = compute_expected(arg0, arg1)
    except ValueError as e:
        print(f"Error computing", file=sys.stderr)
        sys.exit(2)

    out_path = Path("expected0.npy")
    save_array(out_path, result)
    print(f"Created {out_path}: shape={result.shape}, dtype={result.dtype}")


if __name__ == "__main__":
    main(sys.argv)
