"""
FiberHMM model I/O module

Handles loading and saving HMM models in multiple formats:
- .json: Human-readable, fully portable (recommended for saving)
- .npz: Numpy archive (supported for loading, backward compat)
- .pickle/.pkl: Legacy pickle format (supported for loading, backward compat)

Also handles conversion of legacy hmmlearn models (MultinomialHMM, CategoricalHMM).

Saving is JSON-only; legacy formats are retained for loading backward compatibility.
"""

import json
import os
import pickle
import warnings
from typing import Any, Dict, Tuple

import numpy as np

from fiberhmm.core.hmm import FiberHMM


# =============================================================================
# Loading functions (all formats supported for backward compatibility)
# =============================================================================

def load_model(filepath: str, normalize: bool = True) -> FiberHMM:
    """
    Load a model from file.

    Supports multiple formats (auto-detected by extension):
    - .npz: Numpy archive (recommended, stable across versions)
    - .json: JSON format (human-readable, fully portable)
    - .pickle/.pkl: Legacy pickle format (may have version issues)

    Also handles:
    - Legacy hmmlearn MultinomialHMM models (any version)
    - Legacy hmmlearn CategoricalHMM models

    Args:
        filepath: Path to model file
        normalize: If True, normalize states so State 0 = accessible (default True)

    Returns:
        FiberHMM model instance
    """
    if filepath.endswith('.npz'):
        model = _load_npz(filepath)
    elif filepath.endswith('.json'):
        model = _load_json(filepath)
    else:
        # Assume pickle (legacy)
        model = _load_pickle(filepath)

    # Normalize states to ensure correct assignment
    if normalize:
        model.normalize_states()

    return model


def _load_npz(filepath: str) -> FiberHMM:
    """Load model from NPZ format."""
    data = np.load(filepath, allow_pickle=False)

    model = FiberHMM(n_states=int(data['n_states']))
    model.startprob_ = data['startprob']
    model.transmat_ = data['transmat']
    model.emissionprob_ = data['emissionprob']

    return model


def _load_json(filepath: str) -> FiberHMM:
    """Load model from JSON format."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    model = FiberHMM(n_states=data['n_states'])
    model.startprob_ = np.array(data['startprob'])
    model.transmat_ = np.array(data['transmat'])
    model.emissionprob_ = np.array(data['emissionprob'])

    return model


def _load_pickle(filepath: str) -> FiberHMM:
    """Load model from pickle format (legacy)."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    # Check if it's already a native FiberHMM
    if isinstance(obj, FiberHMM):
        return obj

    # Check if it's a dict serialization
    if isinstance(obj, dict):
        if obj.get('model_type') == 'FiberHMM_native':
            return FiberHMM.from_dict(obj)
        # Also handle the new metadata format from train_model.py
        if 'model' in obj:
            inner = obj['model']
            if isinstance(inner, FiberHMM):
                return inner
            if isinstance(inner, dict) and inner.get('model_type') == 'FiberHMM_native':
                return FiberHMM.from_dict(inner)

    # Try to convert from hmmlearn model
    return _convert_hmmlearn_model(obj)


def _convert_hmmlearn_model(hmmlearn_model) -> FiberHMM:
    """
    Convert an hmmlearn model to native FiberHMM.

    Supports:
    - MultinomialHMM (old hmmlearn < 0.3)
    - CategoricalHMM (hmmlearn >= 0.3)
    """
    model = FiberHMM(n_states=2)

    # Extract parameters - these attribute names are consistent across versions
    if hasattr(hmmlearn_model, 'startprob_'):
        model.startprob_ = np.array(hmmlearn_model.startprob_)

    if hasattr(hmmlearn_model, 'transmat_'):
        model.transmat_ = np.array(hmmlearn_model.transmat_)

    # Emission probabilities - different names in different versions
    if hasattr(hmmlearn_model, 'emissionprob_'):
        model.emissionprob_ = np.array(hmmlearn_model.emissionprob_)
    elif hasattr(hmmlearn_model, 'emissionprob'):
        model.emissionprob_ = np.array(hmmlearn_model.emissionprob)

    # Validate
    if model.startprob_ is None or model.transmat_ is None or model.emissionprob_ is None:
        raise ValueError(
            f"Could not extract HMM parameters from {type(hmmlearn_model).__name__}. "
            "Ensure the model has startprob_, transmat_, and emissionprob_ attributes."
        )

    return model


# =============================================================================
# Saving functions (JSON only; legacy formats removed)
# =============================================================================

def save_model(model: FiberHMM, filepath: str,
               context_size: int = 3, mode: str = 'pacbio-fiber'):
    """
    Save model to file in JSON format.

    Only JSON is supported for saving. For backward compatibility when loading,
    see load_model() which supports .npz and .pickle/.pkl as well.

    If the filepath does not end in .json, the extension is replaced with .json
    and a warning is issued.

    Args:
        model: FiberHMM model
        filepath: Output path (.json recommended)
        context_size: Context size used for training (saved as metadata)
        mode: Analysis mode used for training (saved as metadata)
    """
    if not filepath.endswith('.json'):
        old_path = filepath
        base, _ = os.path.splitext(filepath)
        filepath = base + '.json'
        warnings.warn(
            f"Only JSON format is supported for saving. "
            f"Saving to '{filepath}' instead of '{old_path}'."
        )

    _save_json(model, filepath, context_size, mode)


def _save_json(model: FiberHMM, filepath: str, context_size: int, mode: str):
    """Save model in JSON format (human-readable, portable)."""
    data = {
        'model_type': 'FiberHMM',
        'version': '2.0',
        'n_states': model.n_states,
        'startprob': model.startprob_.tolist(),
        'transmat': model.transmat_.tolist(),
        'emissionprob': model.emissionprob_.tolist(),
        'context_size': context_size,
        'mode': mode
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Loading with metadata
# =============================================================================

def load_model_with_metadata(filepath: str, normalize: bool = True) -> Tuple[FiberHMM, int, str]:
    """
    Load model and extract metadata.

    Args:
        filepath: Path to model file
        normalize: If True, normalize states so State 0 = accessible (default True)

    Returns:
        (model, context_size, mode)
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=False)
        model = FiberHMM(n_states=int(data['n_states']))
        model.startprob_ = data['startprob']
        model.transmat_ = data['transmat']
        model.emissionprob_ = data['emissionprob']

        context_size = int(data.get('context_size', 3))
        mode = str(data.get('mode', 'pacbio-fiber'))

    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)

        model = FiberHMM(n_states=data['n_states'])
        model.startprob_ = np.array(data['startprob'])
        model.transmat_ = np.array(data['transmat'])
        model.emissionprob_ = np.array(data['emissionprob'])

        context_size = data.get('context_size', 3)
        mode = data.get('mode', 'pacbio-fiber')

    else:
        # Pickle format
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and 'model' in obj:
            model = obj['model']
            if not isinstance(model, FiberHMM):
                model = _convert_hmmlearn_model(model)
            context_size = obj.get('context_size', 3)
            mode = obj.get('mode', 'pacbio-fiber')
        elif isinstance(obj, FiberHMM):
            model = obj
            context_size = 3
            mode = 'pacbio-fiber'
        else:
            model = _convert_hmmlearn_model(obj)
            context_size = 3
            mode = 'pacbio-fiber'

    # Normalize states to ensure correct assignment
    if normalize:
        model.normalize_states()

    # Normalize old mode names to new names
    mode_aliases = {
        'nanopore': 'nanopore-fiber',
        'stranded': 'nanopore-fiber',
        'pacbio': 'pacbio-fiber',
        'm6a': 'pacbio-fiber',
    }
    if mode in mode_aliases:
        mode = mode_aliases[mode]

    return model, context_size, mode
