"""Indexed, spatial-axis-free concept interaction results."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ...types import Optional, Tuple


@dataclass(frozen=True, eq=False)
class ConceptInteractionResult:
    """Singleton and pairwise scores for one encoded input.

    Parameters
    ----------
    n_concepts
        Ambient number of concept channels.
    active_ids
        Ascending ambient IDs of active channels, shape (d,), int64.
    main_effects
        Singleton scores aligned with active_ids, shape (d,), float32.
    pair_indices
        Evaluated ambient pairs (i, j), shape (P, 2), int64.
    interaction_scores
        Pair scores aligned with pair_indices, shape (P,), float32.
        Inactive requested pairs are zero; unrequested pairs are absent.

    Notes
    -----
    The meaning of the scores is defined by the explainer producing the result
    (e.g. SparseHSIC reports unsigned HSIC dependence components).
    Tensor-valued fields should be compared individually; value equality of
    entire result objects is intentionally undefined.
    """

    n_concepts: int
    active_ids: tf.Tensor
    main_effects: tf.Tensor
    pair_indices: tf.Tensor
    interaction_scores: tf.Tensor


def _validate_pairs(pairs, n_concepts: int) -> Optional[np.ndarray]:
    """Return requested ambient pairs as int64 (P, 2), or None for all active pairs."""
    if pairs is None:
        return None
    requested = np.asarray(pairs)
    if requested.ndim != 2 or requested.shape[1] != 2:
        raise ValueError("pairs must be an integer array of shape (P, 2).")
    if not np.issubdtype(requested.dtype, np.integer) or np.issubdtype(requested.dtype, np.bool_):
        raise ValueError("pairs must contain integer indices, excluding booleans.")
    if np.any(requested[:, 0] < 0) or np.any(requested[:, 1] >= n_concepts):
        raise ValueError("pairs must satisfy 0 <= i < j < n_concepts.")
    if np.any(requested[:, 0] >= requested[:, 1]):
        raise ValueError("pairs must satisfy 0 <= i < j < n_concepts.")
    if len(np.unique(requested, axis=0)) != len(requested):
        raise ValueError("pairs must not contain duplicates.")
    return requested.astype(np.int64)


def _resolve_pairs(
    active: np.ndarray, requested: Optional[np.ndarray], n_concepts: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map ambient pairs to local active-channel indices.

    Parameters
    ----------
    active
        Ascending ambient IDs of the active channels, shape (d,).
    requested
        Validated ambient pairs (P, 2), or None for every distinct active pair.
    n_concepts
        Ambient number of concept channels.

    Returns
    -------
    ambient_pairs
        Reported ambient pairs (P, 2), int64.
    local_pairs
        Pairs indexing the active channels (P, 2); -1 marks an inactive channel.
    valid_rows
        Rows of local_pairs whose two channels are both active.
    """
    if requested is None:
        # Automatic pairs are lexicographic since active IDs are ascending.
        local_pairs = np.stack(np.triu_indices(len(active), k=1), axis=1).astype(np.int64)
        return active[local_pairs].astype(np.int64), local_pairs, np.arange(len(local_pairs))
    ambient_to_local = np.full(n_concepts, -1, np.int64)
    ambient_to_local[active] = np.arange(len(active))
    local_pairs = ambient_to_local[requested]
    return requested, local_pairs, np.flatnonzero(np.all(local_pairs >= 0, axis=1))
