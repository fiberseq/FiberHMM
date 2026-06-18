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
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fiberhmm.core.hmm import FiberHMM

DEFAULT_CONTEXT_SIZE = 3
DEFAULT_MODE = 'pacbio-fiber'
MODE_ALIASES = {
    'nanopore': 'nanopore-fiber',
    'stranded': 'nanopore-fiber',
    'pacbio': 'pacbio-fiber',
    'm6a': 'pacbio-fiber',
}


@dataclass(frozen=True)
class _ModelMetadata:
    context_size: int
    mode: str


@dataclass(frozen=True)
class _LoadedModel:
    model: FiberHMM
    context_size: int
    mode: str

    def as_tuple(self) -> Tuple[FiberHMM, int, str]:
        return self.model, self.context_size, self.mode


@dataclass(frozen=True)
class _JsonSavePath:
    filepath: str
    old_path: str


@dataclass(frozen=True)
class _ModelJsonRecord:
    model_type: str
    version: str
    n_states: int
    startprob: list
    transmat: list
    emissionprob: list
    context_size: int
    mode: str

    def as_dict(self) -> dict:
        return {
            'model_type': self.model_type,
            'version': self.version,
            'n_states': self.n_states,
            'startprob': self.startprob,
            'transmat': self.transmat,
            'emissionprob': self.emissionprob,
            'context_size': self.context_size,
            'mode': self.mode,
        }


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
        normalize: If True, normalize states so State 0 = footprint (default True)

    Returns:
        FiberHMM model instance
    """
    loaded = _load_model_and_metadata_by_extension(filepath)
    return _prepare_loaded_model(loaded.model, normalize)


def _normalize_model_if_requested(model: FiberHMM, normalize: bool) -> FiberHMM:
    if normalize:
        model.normalize_states()
    return model


def _call_model_hook_if_present(model: FiberHMM, hook_name: str) -> FiberHMM:
    hook = getattr(model, hook_name, None)
    if hook is not None:
        hook()
    return model


def _unfreeze_model_logs(model: FiberHMM) -> FiberHMM:
    return _call_model_hook_if_present(model, 'unfreeze_log_probs')


def _prepare_loaded_model(model: FiberHMM, normalize: bool) -> FiberHMM:
    _normalize_model_if_requested(model, normalize)
    return _unfreeze_model_logs(model)


def freeze_model_for_inference(model: FiberHMM) -> FiberHMM:
    """Prepare a loaded model for read-only inference loops."""
    return _call_model_hook_if_present(model, 'freeze_log_probs')


def load_model_for_inference(filepath: str, normalize: bool = True) -> FiberHMM:
    """Load and freeze a model for read-only inference loops."""
    return freeze_model_for_inference(load_model(filepath, normalize=normalize))


def _model_file_format(filepath: str) -> str:
    filepath = os.fspath(filepath)
    normalized_path = filepath.lower()
    if normalized_path.endswith('.npz'):
        return 'npz'
    if normalized_path.endswith('.json'):
        return 'json'
    return 'pickle'


def _load_model_and_metadata_by_extension(filepath: str) -> _LoadedModel:
    file_format = _model_file_format(filepath)
    if file_format == 'npz':
        return _load_npz_with_metadata(filepath)
    if file_format == 'json':
        return _load_json_with_metadata(filepath)
    # Assume pickle (legacy)
    return _load_pickle_with_metadata(filepath)


def _model_from_arrays(n_states, startprob, transmat, emissionprob) -> FiberHMM:
    model = FiberHMM(n_states=int(n_states))
    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)
    model.emissionprob_ = np.array(emissionprob)
    return model


def _model_from_mapping(data) -> FiberHMM:
    return _model_from_arrays(
        data['n_states'],
        data['startprob'],
        data['transmat'],
        data['emissionprob'],
    )


def _mapping_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _metadata_from_mapping(data) -> _ModelMetadata:
    context_size = _mapping_value(data.get('context_size', DEFAULT_CONTEXT_SIZE))
    if context_size is None:
        context_size = DEFAULT_CONTEXT_SIZE
    mode = _mapping_value(data.get('mode', DEFAULT_MODE))
    if mode is not None:
        mode = str(mode)
    return _ModelMetadata(int(context_size), mode)


def _model_and_metadata_from_mapping(data) -> _LoadedModel:
    model = _model_from_mapping(data)
    metadata = _metadata_from_mapping(data)
    return _LoadedModel(model, metadata.context_size, metadata.mode)


def _read_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def _read_pickle(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _coerce_loaded_model(obj) -> FiberHMM:
    if isinstance(obj, FiberHMM):
        return obj
    if isinstance(obj, dict) and obj.get('model_type') == 'FiberHMM_native':
        return FiberHMM.from_dict(obj)
    return _convert_hmmlearn_model(obj)


def _normalize_mode(mode) -> str:
    if mode is None:
        return DEFAULT_MODE
    mode_name = str(mode).strip()
    return MODE_ALIASES.get(mode_name, mode_name)


def _load_npz(filepath: str) -> FiberHMM:
    """Load model from NPZ format."""
    return _load_npz_with_metadata(filepath).model


def _load_npz_with_metadata(filepath: str) -> _LoadedModel:
    """Load model and metadata from NPZ format."""
    with np.load(filepath, allow_pickle=False) as data:
        return _model_and_metadata_from_mapping(data)


def _load_json(filepath: str) -> FiberHMM:
    """Load model from JSON format."""
    return _load_json_with_metadata(filepath).model


def _load_json_with_metadata(filepath: str) -> _LoadedModel:
    """Load model and metadata from JSON format."""
    data = _read_json(filepath)
    return _model_and_metadata_from_mapping(data)


def _load_pickle(filepath: str) -> FiberHMM:
    """Load model from pickle format (legacy)."""
    return _load_pickle_with_metadata(filepath).model


def _pickle_payload_has_metadata(obj) -> bool:
    return isinstance(obj, dict) and 'model' in obj


def _load_pickle_with_metadata(filepath: str) -> _LoadedModel:
    """Load model and metadata from pickle format (legacy)."""
    obj = _read_pickle(filepath)
    if _pickle_payload_has_metadata(obj):
        model = _coerce_loaded_model(obj['model'])
        metadata = _metadata_from_mapping(obj)
        return _LoadedModel(model, metadata.context_size, metadata.mode)
    return _LoadedModel(_coerce_loaded_model(obj), DEFAULT_CONTEXT_SIZE, DEFAULT_MODE)


def _model_attr_array(obj, *attr_names):
    for attr_name in attr_names:
        if hasattr(obj, attr_name):
            return np.array(getattr(obj, attr_name))
    return None


def _convert_hmmlearn_model(hmmlearn_model) -> FiberHMM:
    """
    Convert an hmmlearn model to native FiberHMM.

    Supports:
    - MultinomialHMM (old hmmlearn < 0.3)
    - CategoricalHMM (hmmlearn >= 0.3)
    """
    model = FiberHMM(n_states=2)

    # Extract parameters - these attribute names are consistent across versions
    model.startprob_ = _model_attr_array(hmmlearn_model, 'startprob_')
    model.transmat_ = _model_attr_array(hmmlearn_model, 'transmat_')
    # Emission probabilities - different names in different versions
    model.emissionprob_ = _model_attr_array(
        hmmlearn_model, 'emissionprob_', 'emissionprob',
    )

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

def _json_save_path(filepath: str) -> _JsonSavePath:
    filepath = os.fspath(filepath)
    if filepath.lower().endswith('.json'):
        return _JsonSavePath(filepath, filepath)
    base, _ = os.path.splitext(filepath)
    return _JsonSavePath(base + '.json', filepath)


def _json_save_redirect_warning(filepath: str, old_path: str) -> str:
    return (
        "Only JSON format is supported for saving. "
        f"Saving to '{filepath}' instead of '{old_path}'."
    )


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
    save_path = _json_save_path(filepath)
    if save_path.old_path != save_path.filepath:
        warnings.warn(
            _json_save_redirect_warning(save_path.filepath, save_path.old_path)
        )

    _save_json(model, save_path.filepath, context_size, mode)


def _model_json_record_value(
    model: FiberHMM,
    context_size: int,
    mode: str,
) -> _ModelJsonRecord:
    return _ModelJsonRecord(
        model_type='FiberHMM',
        version='2.0',
        n_states=model.n_states,
        startprob=model.startprob_.tolist(),
        transmat=model.transmat_.tolist(),
        emissionprob=model.emissionprob_.tolist(),
        context_size=context_size,
        mode=mode,
    )


def _model_json_record(model: FiberHMM, context_size: int, mode: str) -> dict:
    return _model_json_record_value(model, context_size, mode).as_dict()


def _save_json(model: FiberHMM, filepath: str, context_size: int, mode: str):
    """Save model in JSON format (human-readable, portable)."""
    data = _model_json_record_value(model, context_size, mode).as_dict()
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
        normalize: If True, normalize states so State 0 = footprint (default True)

    Returns:
        (model, context_size, mode)
    """
    loaded = _load_model_and_metadata_by_extension(filepath)
    model, context_size, mode = loaded.as_tuple()

    # Normalize old mode names to new names
    mode = _normalize_mode(mode)

    return (
        _prepare_loaded_model(model, normalize),
        context_size,
        mode,
    )
