"""Indexed, spatial-axis-free concept interaction results."""

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True, eq=False)
class ConceptInteractionResult:
    """Singleton and pairwise dependence for one encoded input.

    Parameters
    ----------
    n_concepts
        Ambient number of concept channels.
    active_ids
        Ascending ambient IDs of active channels, shape (d,), int64.
    main_effects
        Singleton HSIC aligned with active_ids, shape (d,), float32.
    pair_indices
        Evaluated ambient pairs (i, j), shape (P, 2), int64.
    interaction_scores
        Pair HSIC aligned with pair_indices, shape (P,), float32.
        Inactive requested pairs are zero; unrequested pairs are absent.

    Notes
    -----
    Tensor-valued fields should be compared individually; value equality of
    entire result objects is intentionally undefined.
    """

    n_concepts: int
    active_ids: tf.Tensor
    main_effects: tf.Tensor
    pair_indices: tf.Tensor
    interaction_scores: tf.Tensor
