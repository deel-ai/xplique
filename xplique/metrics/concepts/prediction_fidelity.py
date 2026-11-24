"""Agreement between original and reconstructed, already-decoded model predictions."""

from collections.abc import Iterable

import numpy as np
from scipy.special import rel_entr

from ...types import Any, Dict, Optional, Tuple
from ._utils import array_input, observation_chunks, validate_batch_size


def _prediction_pair(batch):
    """Reject accidental iteration over a bare pair of prediction arrays."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError(
            "Expected an iterable of pairs: [(original, reconstructed), ...], "
            "not a bare (original, reconstructed) pair."
        )
    return batch


def _mean(total, count):
    return float(total / count) if count else np.nan


def _log_probabilities(logits):
    with np.errstate(over="ignore"):
        shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _classification_kl(original, reconstructed, prediction_format):
    if prediction_format == "probabilities":
        for predictions in (original, reconstructed):
            if np.any((predictions < 0) | (predictions > 1)) or not np.allclose(
                predictions.sum(axis=1), 1.0, rtol=0, atol=1e-6
            ):
                raise ValueError("Probabilities must be in [0, 1] and each row must sum to one.")
        # Remove normalization roundoff in supplied probabilities (e.g. float32 softmax).
        original = original / original.sum(axis=1, keepdims=True)
        reconstructed = reconstructed / reconstructed.sum(axis=1, keepdims=True)
        terms = rel_entr(original, reconstructed)
    else:
        log_original = _log_probabilities(original)
        log_reconstructed = _log_probabilities(reconstructed)
        original_probs = np.exp(log_original)
        terms = np.zeros_like(original_probs)
        support = (original_probs > 0) & np.isfinite(log_reconstructed)
        terms[support] = original_probs[support] * (
            log_original[support] - log_reconstructed[support]
        )
        # A finite logit gap can exceed float64 even when its weighted term
        # does not. In that case the log normalizer is negligible at the
        # gap's scale; halve before subtracting, then weight before doubling.
        rows, cols = np.nonzero((original_probs > 0) & ~np.isfinite(log_reconstructed))
        if rows.size:
            half_gap = reconstructed.max(axis=1)[rows] / 2 - reconstructed[rows, cols] / 2
            with np.errstate(over="ignore"):
                terms[rows, cols] = (original_probs[rows, cols] * half_gap) * 2
    with np.errstate(over="ignore"):
        return np.maximum(terms.sum(axis=1), 0.0)


class ClassificationPredictionFidelity:
    """Compare original and reconstructed predictions for single-label classification.

    Parameters
    ----------
    prediction_format : {'logits', 'probabilities'}
        How to interpret both sets of predictions. Probabilities must sum to one
        per row (absolute tolerance ``1e-6``); they are renormalized before KL.
    batch_size : int or None
        Maximum number of examples per calculation chunk. ``None`` (default)
        processes each supplied array at once.

    Notes
    -----
    All scores measure prediction-to-prediction agreement, not ground-truth accuracy.
    Positive original probability with zero reconstructed probability yields infinite
    directional KL divergence. Prefer logits when available to avoid float32 softmax
    underflow to zero.
    """

    def __init__(self, *, prediction_format: str, batch_size: Optional[int] = None):
        if prediction_format not in ("logits", "probabilities"):
            raise ValueError("prediction_format must be 'logits' or 'probabilities'.")
        self.prediction_format = prediction_format
        self.batch_size = validate_batch_size(batch_size)

    def evaluate(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
        """Return pooled top-1 agreement and directional KL (original to reconstructed).

        Parameters
        ----------
        original : np.ndarray
            Finite real array of shape ``(N, C)`` with ``N >= 1`` and ``C >= 2``.
        reconstructed : np.ndarray
            Finite real array with the exact same shape, example order, and class
            order as ``original``.

        Returns
        -------
        report : dict
            Dictionary containing ``top1_agreement`` (float), ``mean_kl_divergence``
            (float, in nats), ``n_agreeing`` (int), and ``n_examples`` (int).
            Top-1 ties use the first class index.

        Raises
        ------
        ValueError
            If arrays differ in shape, are not 2-D with ``C >= 2``, are empty or
            nonfinite, or violate probability bounds/normalization when
            ``prediction_format='probabilities'``.
        """
        return self.evaluate_batches(((original, reconstructed),))

    def __call__(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
        """Alias for ``evaluate``."""
        return self.evaluate(original, reconstructed)

    def evaluate_batches(self, batches: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Pool examples from a one-pass iterable of paired prediction arrays.

        Parameters
        ----------
        batches : iterable of (np.ndarray, np.ndarray)
            One-pass iterable yielding ``(original, reconstructed)`` tuple/list pairs
            satisfying ``evaluate`` with a fixed class count ``C`` across batches.
            Pass ``[(original, reconstructed)]``, not a bare ``(original, reconstructed)``
            pair.

        Returns
        -------
        report : dict
            Same report as ``evaluate``. Individual examples have equal weight,
            regardless of the size of the batch that contains them.

        Raises
        ------
        ValueError
            If any item is not a length-2 tuple/list, the stream is empty, class
            counts change between batches, or an array violates ``evaluate`` rules.
        """
        n_examples = n_agreeing = 0
        mean_kl = 0.0
        n_classes = None
        for batch in batches:
            original, reconstructed = _prediction_pair(batch)
            original = array_input(original, "original")
            reconstructed = array_input(reconstructed, "reconstructed")
            if original.shape != reconstructed.shape or original.ndim != 2 or original.shape[1] < 2:
                raise ValueError("Predictions must have the same shape (N, C), with C >= 2.")
            if n_classes is not None and original.shape[1] != n_classes:
                raise ValueError("Class count cannot change between batches.")
            n_classes = original.shape[1]
            for left, right in zip(
                observation_chunks(original, self.batch_size),
                observation_chunks(reconstructed, self.batch_size),
            ):
                kl = _classification_kl(left, right, self.prediction_format)
                n_agreeing += int(np.count_nonzero(left.argmax(axis=1) == right.argmax(axis=1)))
                n_examples += len(left)
                if np.isinf(mean_kl) or np.isinf(kl).any():
                    mean_kl = np.inf
                else:
                    # Scale before summing to avoid overflowing on finite large KL scores.
                    weight = len(kl) / n_examples
                    chunk_mean = float(np.sum(kl / len(kl)))
                    mean_kl += weight * (chunk_mean - mean_kl)
        if not n_examples:
            raise ValueError("An evaluation must contain predictions.")
        return {
            "top1_agreement": _mean(n_agreeing, n_examples),
            "mean_kl_divergence": mean_kl,
            "n_agreeing": n_agreeing,
            "n_examples": n_examples,
        }
