"""Agreement between original and reconstructed, already-decoded model predictions."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import rel_entr

from ...types import Any, Dict, Optional, Tuple, Union
from ._utils import array_input, is_real_scalar, observation_chunks, validate_batch_size


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


def _threshold(value, name, *, positive=False) -> float:
    if (
        not is_real_scalar(value)
        or not np.isfinite(value)
        or not 0 <= value <= 1
        or (positive and value == 0)
    ):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}.")
    return float(value)


def _real_array(values, name):
    """Validate finite real values, allowing empty instance arrays."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numerical array.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values.")
    return values


def _score_mask(scores, n_instances, threshold):
    scores = _real_array(scores, "scores")
    if scores.shape != (n_instances,) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must have one value in [0, 1] per instance.")
    return scores >= threshold


def _eligible_matches(similarity, eligible):
    """Select as many eligible matches as possible, then maximize overlap."""
    if not eligible.any():
        return 0, 0.0
    # Each overlap is in [0, 1], so one extra eligible edge outranks any
    # difference in total overlap. Ineligible edges only fill the assignment.
    bonus = 1 + min(similarity.shape)
    rows, cols = linear_sum_assignment(np.where(eligible, bonus + similarity, 0), maximize=True)
    matched = eligible[rows, cols]
    return int(matched.sum()), float(similarity[rows[matched], cols[matched]].sum())


def _instance_report(batches, score_image, overlap_name) -> Dict[str, Any]:
    """Pool matched pairs and instance counts without averaging per-image F1."""
    sentinel = object()
    n_images = n_original = n_reconstructed = n_matched = 0
    overlap_sum = 0.0
    for batch in batches:
        original, reconstructed = _prediction_pair(batch)
        if not isinstance(original, Iterable) or not isinstance(reconstructed, Iterable):
            raise ValueError(
                "Original and reconstructed inputs must be iterables of per-image predictions."
            )
        for left, right in zip_longest(original, reconstructed, fillvalue=sentinel):
            if left is sentinel or right is sentinel:
                raise ValueError("Original and reconstructed sets must contain the same images.")
            left_count, right_count, count, overlap = score_image(left, right)
            n_images += 1
            n_original += left_count
            n_reconstructed += right_count
            n_matched += count
            overlap_sum += overlap
    if not n_images:
        raise ValueError("An evaluation must contain at least one image.")
    return {
        "precision": _mean(n_matched, n_reconstructed),
        "recall": _mean(n_matched, n_original),
        "f1": _mean(2 * n_matched, n_original + n_reconstructed),
        overlap_name: _mean(overlap_sum, n_matched),
        "n_matched": n_matched,
        "n_original": n_original,
        "n_reconstructed": n_reconstructed,
        "n_images": n_images,
    }


@dataclass(eq=False)
class BoxPredictions:
    """One image's decoded detections as ``xyxy`` boxes, class IDs, and scores.

    Parameters
    ----------
    boxes : np.ndarray
        Finite real ``xyxy`` coordinates of shape ``(instances, 4)`` with
        strictly positive width and height (``x2 > x1`` and ``y2 > y1``).
        An empty image uses shape ``(0, 4)``.
    class_ids : np.ndarray
        Nonnegative integer class IDs of shape ``(instances,)`` fitting in
        ``int64``. An empty image uses an integer array of shape ``(0,)``.
    scores : np.ndarray
        Finite confidence scores in ``[0, 1]`` of shape ``(instances,)``.

    Notes
    -----
    Instances compare by identity (``eq=False``) to avoid ambiguous NumPy array
    truth values. Compare individual array fields with NumPy testing utilities.
    """

    boxes: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray


def _retained_boxes(predictions, confidence_threshold):
    if not isinstance(predictions, BoxPredictions):
        raise ValueError("Each image must provide BoxPredictions.")
    boxes = _real_array(predictions.boxes, "boxes")
    class_ids = np.asarray(predictions.class_ids)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must have shape (instances, 4) and positive xyxy extents.")
    if (
        class_ids.shape != (len(boxes),)
        or class_ids.dtype.kind not in "iu"
        or np.any(class_ids < 0)
        or np.any(class_ids > np.iinfo(np.int64).max)
    ):
        raise ValueError("class_ids must have one nonnegative integer ID per box.")
    retain = _score_mask(predictions.scores, len(boxes), confidence_threshold)
    return boxes[retain], class_ids[retain].astype(np.int64, copy=False)


def _box_iou_matrix(left, right):
    # Equivalent to the attribution IoU geometry, without its epsilon, so
    # identical positive-area boxes score exactly 1 at a threshold of 1.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        left_area = np.prod(left[:, 2:] - left[:, :2], axis=1)
        right_area = np.prod(right[:, 2:] - right[:, :2], axis=1)
        start = np.maximum(left[:, None, :2], right[None, :, :2])
        end = np.minimum(left[:, None, 2:], right[None, :, 2:])
        intersection = np.maximum(end - start, 0).prod(axis=2)
        union = left_area[:, None] - intersection + right_area[None, :]
    if not np.isfinite(union).all() or np.any(left_area <= 0) or np.any(right_area <= 0):
        raise ValueError("Box areas must be representable and positive; rescale coordinates.")
    return np.clip(intersection / union, 0.0, 1.0)


class DetectionPredictionFidelity:
    """Compare decoded detections with same-class, one-to-one IoU matching.

    Parameters
    ----------
    iou_threshold : float
        Minimum match IoU, inclusive, in ``(0, 1]``. Default ``0.5``.
    confidence_threshold : float
        Minimum confidence on either side, inclusive, in ``[0, 1]``.
        Default ``0.0`` retains all supplied detections.
    """

    def __init__(self, *, iou_threshold: float = 0.5, confidence_threshold: float = 0.0):
        self.iou_threshold = _threshold(iou_threshold, "iou_threshold", positive=True)
        self.confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")

    def _score_image(self, original, reconstructed):
        left, left_class = _retained_boxes(original, self.confidence_threshold)
        right, right_class = _retained_boxes(reconstructed, self.confidence_threshold)
        if not len(left) or not len(right):
            return len(left), len(right), 0, 0.0
        overlaps = _box_iou_matrix(left, right)
        eligible = (overlaps >= self.iou_threshold) & (left_class[:, None] == right_class[None, :])
        count, total = _eligible_matches(overlaps, eligible)
        return len(left), len(right), count, total

    def evaluate(
        self, original: Iterable[BoxPredictions], reconstructed: Iterable[BoxPredictions]
    ) -> Dict[str, Any]:
        """Compare two collections of decoded boxes for the same ordered images.

        Parameters
        ----------
        original : iterable of BoxPredictions
            Unpadded original detections in common coordinates and class order.
            An empty image has arrays of shapes ``(0, 4)``, ``(0,)``, and ``(0,)``.
        reconstructed : iterable of BoxPredictions
            Unpadded reconstructed detections aligned image-for-image with ``original``.

        Returns
        -------
        report : dict
            Pooled ``precision``, ``recall``, ``f1``, and ``mean_matched_iou`` (floats),
            with ``n_matched``, ``n_original``, ``n_reconstructed``, and ``n_images``
            (ints). A zero denominator yields ``NaN``. These scores compare predictions
            to predictions; they are not average precision.

        Raises
        ------
        ValueError
            If inputs are not aligned iterables of ``BoxPredictions``, contain zero
            images, have non-positive/overflowing box areas, invalid class IDs, or
            scores outside ``[0, 1]``.
        """
        return self.evaluate_batches(((original, reconstructed),))

    def __call__(
        self, original: Iterable[BoxPredictions], reconstructed: Iterable[BoxPredictions]
    ) -> Dict[str, Any]:
        """Alias for ``evaluate``."""
        return self.evaluate(original, reconstructed)

    def evaluate_batches(
        self, batches: Iterable[Tuple[Iterable[BoxPredictions], Iterable[BoxPredictions]]]
    ) -> Dict[str, Any]:
        """Pool the ``evaluate`` report from a one-pass stream of paired image batches.

        Parameters
        ----------
        batches : iterable of (iterable of BoxPredictions, iterable of BoxPredictions)
            One-pass iterable of ``(original_batch, reconstructed_batch)`` tuple/list
            pairs. Pass ``[(original, reconstructed)]``, not a bare
            ``(original, reconstructed)`` pair.

        Returns
        -------
        report : dict
            Pooled detection agreement report with the same keys and semantics as
            ``evaluate``, without averaging per-batch or per-image F1.

        Raises
        ------
        ValueError
            If any item is not a length-2 tuple/list, no images are seen, batch lengths
            differ between sides, or any image violates ``evaluate`` rules.
        """
        return _instance_report(batches, self._score_image, "mean_matched_iou")


@dataclass(eq=False)
class PosePredictions:
    """One image's decoded poses in a single skeleton, with instance scores.

    Parameters
    ----------
    keypoints : np.ndarray
        Real coordinates of shape ``(instances, joints, 2)``.
    scores : np.ndarray
        One finite confidence per instance, in ``[0, 1]``.
    areas : np.ndarray or None
        Positive object areas in squared coordinate units, required on the
        original side only.
    valid_joints : np.ndarray or None
        Boolean mask of shape ``(instances, joints)``, required on the original
        side only. At least one joint must be valid per retained original pose.
        On the reconstructed side neither this field nor ``areas`` is used.

    Notes
    -----
    Instances compare by identity (``eq=False``) to avoid ambiguous NumPy array
    truth values. Compare individual array fields with NumPy testing utilities.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    areas: Optional[np.ndarray] = None
    valid_joints: Optional[np.ndarray] = None


def _retained_poses(predictions, n_joints, confidence_threshold, *, reference=False):
    if not isinstance(predictions, PosePredictions):
        raise ValueError("Each image must provide PosePredictions.")
    joints = np.asarray(predictions.keypoints)
    if joints.ndim != 3 or joints.shape[1:] != (n_joints, 2):
        raise ValueError("keypoints must have shape (instances, joints, 2), matching sigmas.")
    if joints.dtype.kind not in "iuf":
        raise ValueError("keypoints must be a real numerical array.")
    retain = _score_mask(predictions.scores, len(joints), confidence_threshold)
    if not reference:
        return _real_array(joints, "reconstructed keypoints")[retain]
    valid = np.asarray(predictions.valid_joints)
    if valid.shape != joints.shape[:2] or valid.dtype.kind != "b":
        raise ValueError("Original valid_joints must be boolean with shape (instances, joints).")
    areas = _real_array(predictions.areas, "original areas")
    if areas.shape != (len(joints),) or np.any(areas <= 0):
        raise ValueError("Original areas must have one positive value per instance.")
    if np.any(~valid[retain].any(axis=1)):
        raise ValueError("Every retained original pose must have at least one valid joint.")
    joints = _real_array(np.where(valid[..., None], joints, 0.0), "valid original keypoints")
    return joints[retain], areas[retain], valid[retain]


def _pose_oks_matrix(left, right, areas, valid, sigmas):
    # Accumulate one joint at a time: memory scales with the number of
    # instance pairs, not with instance pairs times skeleton size.
    overlaps = np.zeros((len(left), len(right)))
    scale = np.sqrt(areas)
    for joint, sigma in enumerate(sigmas):
        rows = valid[:, joint]
        with np.errstate(over="ignore", under="ignore"):
            distance = left[rows, joint, None, :] - right[None, :, joint, :]
            distance /= scale[rows, None, None]
            distance = (distance / 2) / sigma
            overlaps[rows] += np.exp(-0.5 * np.square(distance).sum(axis=2))
    return overlaps / valid.sum(axis=1, keepdims=True)


class PosePredictionFidelity:
    """Compare decoded poses for one skeleton using one-to-one OKS matching.

    Parameters
    ----------
    sigmas : array-like of float
        Positive per-joint OKS sigmas, in the same order as the joint axis. The
        COCO convention is used: the distance scale contains ``2 * sigma``.
    oks_threshold : float
        Minimum match OKS, inclusive, in ``(0, 1]``. Default ``0.5``.
    confidence_threshold : float
        Minimum instance confidence on either side, inclusive, in ``[0, 1]``.
        Default ``0.0`` retains all supplied poses.
    """

    def __init__(
        self,
        *,
        sigmas: Union[np.ndarray, Iterable[float]],
        oks_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
    ):
        sigmas_arr = _real_array(sigmas, "sigmas")
        if sigmas_arr.ndim != 1 or not sigmas_arr.size or np.any(sigmas_arr <= 0):
            raise ValueError("sigmas must be a nonempty vector of positive values.")
        self.sigmas: Tuple[float, ...] = tuple(sigmas_arr.tolist())
        self.oks_threshold = _threshold(oks_threshold, "oks_threshold", positive=True)
        self.confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")

    def _score_image(self, original, reconstructed):
        left, areas, valid = _retained_poses(
            original, len(self.sigmas), self.confidence_threshold, reference=True
        )
        right = _retained_poses(reconstructed, len(self.sigmas), self.confidence_threshold)
        if not len(left) or not len(right):
            return len(left), len(right), 0, 0.0
        overlap = _pose_oks_matrix(left, right, areas, valid, self.sigmas)
        count, total = _eligible_matches(overlap, overlap >= self.oks_threshold)
        return len(left), len(right), count, total

    def evaluate(
        self, original: Iterable[PosePredictions], reconstructed: Iterable[PosePredictions]
    ) -> Dict[str, Any]:
        """Compare two collections of decoded poses for the same ordered images.

        Parameters
        ----------
        original : iterable of PosePredictions
            Original poses with positive ``areas`` and boolean ``valid_joints`` masks.
        reconstructed : iterable of PosePredictions
            Reconstructed poses aligned image-for-image with ``original`` in the same
            coordinate system and skeleton order. Only original ``areas`` and
            ``valid_joints`` determine the OKS scale and scored joint set.

        Returns
        -------
        report : dict
            Pooled ``precision``, ``recall``, ``f1``, and ``mean_matched_oks`` (floats)
            with the same denominators and empty-case conventions as
            ``DetectionPredictionFidelity``. This is not keypoint AP.

        Raises
        ------
        ValueError
            If inputs are not aligned iterables of ``PosePredictions``, contain zero
            images, have mismatched skeleton widths, non-real/nonfinite coordinates,
            non-positive original areas, or a retained original pose with no valid joints.
        """
        return self.evaluate_batches(((original, reconstructed),))

    def __call__(
        self, original: Iterable[PosePredictions], reconstructed: Iterable[PosePredictions]
    ) -> Dict[str, Any]:
        """Alias for ``evaluate``."""
        return self.evaluate(original, reconstructed)

    def evaluate_batches(
        self, batches: Iterable[Tuple[Iterable[PosePredictions], Iterable[PosePredictions]]]
    ) -> Dict[str, Any]:
        """Pool the ``evaluate`` report from a one-pass stream of paired image batches.

        Parameters
        ----------
        batches : iterable of (iterable of PosePredictions, iterable of PosePredictions)
            One-pass iterable of ``(original_batch, reconstructed_batch)`` tuple/list
            pairs. Pass ``[(original, reconstructed)]``, not a bare
            ``(original, reconstructed)`` pair.

        Returns
        -------
        report : dict
            Pooled pose agreement report with the same keys and semantics as
            ``evaluate``, without averaging per-batch or per-image F1.

        Raises
        ------
        ValueError
            If any item is not a length-2 tuple/list, no images are seen, batch lengths
            differ between sides, or any image violates ``evaluate`` rules.
        """
        return _instance_report(batches, self._score_image, "mean_matched_oks")
