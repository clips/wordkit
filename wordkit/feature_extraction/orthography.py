"""Functions for extracting orthographical features."""
import numpy as np


def one_hot_characters(characters):
    """
    Extract one-hot encoded binary vectors for your characters.

    Each character will be assigned a unique one-hot encoded binary
    vector. Words transformed using this feature set will thus be
    matrices of one-hot encoded vectors.

    Parameters
    ----------
    characters : string or list
        The unique characters occurring in your dataset.

    Returns
    -------
    features : dict
        A dictionary mapping from characters to one-hot encoded arrays.

    """
    binary_features = np.eye(len(characters))
    return {l: binary_features[idx]
            for idx, l in enumerate(characters)}
