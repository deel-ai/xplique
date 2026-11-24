"""Analytical and integration tests for downstream reconstruction agreement."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import softmax

from xplique.metrics.concepts import (
    BoxPredictions,
    ClassificationPredictionFidelity,
    DetectionPredictionFidelity,
    PosePredictionFidelity,
    PosePredictions,
    ReconstructionR2,
)
from xplique.metrics.concepts.prediction_fidelity import _eligible_matches


def test_classification_probabilities_direction_and_zero_support():
    metric = ClassificationPredictionFidelity(prediction_format="probabilities")
    original = [[1.0, 0.0], [0.5, 0.5]]
    reconstructed = [[0.25, 0.75], [0.5, 0.5]]
    report = metric(original, reconstructed)
    assert report == {
        "top1_agreement": 0.5,
        "mean_kl_divergence": pytest.approx(-np.log(0.25) / 2),
        "n_agreeing": 1,
        "n_examples": 2,
    }
    assert np.isinf(metric(reconstructed, original)["mean_kl_divergence"])
    assert metric(original, original)["mean_kl_divergence"] == 0
    with pytest.raises(ValueError, match="Probabilities"):
        metric(original, [[0.2, 0.2], [0.5, 0.5]])


@pytest.mark.parametrize("size", [None, 1, 2])
def test_classification_logits_streaming_and_validation(size):
    left = np.array([[1000, -1000], [2, 1], [1, 2], [-1, 2]])
    right = np.array([[-1000, 1000], [2, 1], [3, 0], [4, 6]])
    metric = ClassificationPredictionFidelity(prediction_format="logits", batch_size=size)
    full = metric(left, right)
    assert metric.evaluate_batches(iter([(left[:3], right[:3]), (left[3:], right[3:])])) == (
        pytest.approx(full)
    )
    assert full["top1_agreement"] == 0.5
    assert np.isfinite(full["mean_kl_divergence"])
    assert metric(left + 512, right - 256) == pytest.approx(full)
    probabilities = ClassificationPredictionFidelity(prediction_format="probabilities")
    assert metric(left[1:], right[1:]) == pytest.approx(
        probabilities(softmax(left[1:], axis=1), softmax(right[1:], axis=1))
    )
    with pytest.raises(ValueError, match="same shape"):
        metric(left, right[:1])
    with pytest.raises(ValueError, match="Class count"):
        metric.evaluate_batches([(left, right), (np.ones((1, 3)), np.ones((1, 3)))])


def test_classification_weights_a_gap_that_exceeds_float64_range():
    metric = ClassificationPredictionFidelity(prediction_format="logits")
    report = metric([[0.0, 0.0]], [[1e308, -1e308]])
    assert report["mean_kl_divergence"] == pytest.approx(1e308)
    assert np.isinf(metric([[-1e308, 1e308]], [[1e308, -1e308]])["mean_kl_divergence"])


def _boxes(boxes, labels, scores=None):
    return BoxPredictions(
        np.asarray(boxes).reshape(-1, 4),
        np.asarray(labels, dtype=int),
        np.ones(len(labels)) if scores is None else np.asarray(scores),
    )


def test_matching_prioritizes_eligible_cardinality_then_overlap():
    overlaps = np.array([[1.0, 0.4], [0.4, 0.0]])
    assert _eligible_matches(overlaps, overlaps >= 0.4) == (2, pytest.approx(0.8))
    overlaps = np.array([[0.8, 0.9], [0.9, 0.8]])
    assert _eligible_matches(overlaps, overlaps >= 0.5) == (2, pytest.approx(1.8))


def test_detection_reordering_duplicates_classes_and_threshold():
    a, b = [0, 0, 2, 2], [8, 8, 10, 10]
    matching_a = [0, 0, 1, 2]  # IoU exactly 0.5.
    left = _boxes([a, b], [0, 1])
    right = _boxes([b, a, matching_a, matching_a], [0, 0, 0, 0], [1.0, 0.49, 0.5, 0.8])
    metric = DetectionPredictionFidelity(confidence_threshold=0.5)
    expected = {
        "precision": 1 / 3,
        "recall": 1 / 2,
        "f1": 2 / 5,
        "mean_matched_iou": 0.5,
        "n_matched": 1,
        "n_original": 2,
        "n_reconstructed": 3,
        "n_images": 1,
    }
    assert metric([left], [right]) == pytest.approx(expected)
    reordered = BoxPredictions(right.boxes[::-1], right.class_ids[::-1], right.scores[::-1])
    assert metric([left], [reordered]) == pytest.approx(expected)
    strict = DetectionPredictionFidelity(iou_threshold=1)
    assert strict([left], [left])["mean_matched_iou"] == 1


def test_detection_empty_images_and_pooled_streaming():
    empty = _boxes([], [])
    hit = _boxes([[0, 0, 2, 2]], [0])
    metric = DetectionPredictionFidelity()
    assert np.isnan(metric([empty], [empty])["f1"])
    one_sided = metric([empty], [hit])
    assert one_sided["f1"] == 0
    assert np.isnan(one_sided["recall"])
    assert one_sided["precision"] == 0
    streamed = metric.evaluate_batches(iter([([hit, hit], [hit, empty]), ([empty], [empty])]))
    pooled = metric([hit, hit, empty], [hit, empty, empty])
    assert streamed == pytest.approx(pooled)
    assert pooled["f1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="same images"):
        metric([empty], [empty, hit])
    with pytest.raises(ValueError, match="extents"):
        metric([_boxes([[0, 0, 0, 1]], [0])], [empty])


def test_pose_uses_only_reference_joint_mask_and_scale():
    original = PosePredictions([[[0.0, 0.0], [np.nan, np.nan]]], [1.0], [25.0], [[True, False]])
    reconstruction = PosePredictions([[[1.0, 0.0], [999.0, 999.0]]], [1.0])
    metric = PosePredictionFidelity(sigmas=[0.1, 0.1])
    expected = np.exp(-0.5)  # 1 pixel / (sqrt(25) * 2 * 0.1).
    assert metric([original], [reconstruction])["mean_matched_oks"] == pytest.approx(expected)
    hidden = replace(reconstruction, valid_joints=[[False, False]], areas=[10000])
    assert metric([original], [hidden])["mean_matched_oks"] == pytest.approx(expected)
    distant = replace(reconstruction, keypoints=[[[2.0, 0.0], [0.0, 0.0]]])
    report = metric([original], [distant])
    assert report["f1"] == 0
    assert np.isnan(report["mean_matched_oks"])
    with pytest.raises(ValueError, match="at least one valid joint"):
        metric([replace(original, valid_joints=[[False, False]])], [reconstruction])


def test_pose_reordering_empty_and_streaming():
    joints = np.array([[[1.0, 2.0]], [[11.0, 2.0]]])
    original = PosePredictions(joints, [1.0, 1.0], [4, 4], [[True], [True]])
    reconstructed = PosePredictions(joints[::-1], [1.0, 1.0])
    empty = PosePredictions(np.empty((0, 1, 2)), [], [], np.empty((0, 1), dtype=bool))
    metric = PosePredictionFidelity(sigmas=[0.1], oks_threshold=1)
    assert metric([original], [reconstructed])["f1"] == 1
    assert np.isnan(metric([empty], [empty])["f1"])
    streamed = metric.evaluate_batches([([empty], [empty]), ([original], [reconstructed])])
    assert streamed == pytest.approx(metric([empty, original], [empty, reconstructed]))


class _AffineFactorizer:
    """Keep one varying coordinate and reconstruct the other from a decoder bias."""

    def encode(self, activations):
        return activations[:, :1]

    def decode(self, codes):
        return np.concatenate((codes, np.full_like(codes, -0.01)), axis=1)


@pytest.mark.parametrize("backend", ["tensorflow", "torch"])
def test_paired_frozen_head_with_affine_decoder(backend):
    factorizer = _AffineFactorizer()
    x = np.array([[1.0, 0.01], [100.0, -0.02]], dtype=np.float32)
    reconstructed = factorizer.decode(factorizer.encode(x))
    weights = np.array([[0, 0], [1, -1]], dtype=np.float32)
    if backend == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        head = tf.keras.layers.Dense(2, use_bias=False, trainable=False)
        head.build(x.shape)
        head.set_weights([weights])
        original_pred = head(x, training=False).numpy()
        identity_pred = head(x.copy(), training=False).numpy()
        reconstructed_pred = head(reconstructed, training=False).numpy()
    else:
        torch = pytest.importorskip("torch")
        head = torch.nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(weights.T))
            original_pred = head(torch.from_numpy(x)).numpy()
            identity_pred = head(torch.from_numpy(x.copy())).numpy()
            reconstructed_pred = head(torch.from_numpy(reconstructed)).numpy()
        assert not head.training
    metric = ClassificationPredictionFidelity(prediction_format="logits")
    assert ReconstructionR2()(x, reconstructed) > 0.999
    assert metric(original_pred, identity_pred)["top1_agreement"] == 1
    assert metric(original_pred, reconstructed_pred)["top1_agreement"] == 0.5
    assert metric(original_pred, reconstructed_pred)["mean_kl_divergence"] > 0


def test_prediction_dataclasses_compare_by_identity():
    box_a = _boxes([[0, 0, 2, 2], [2, 2, 4, 4]], [0, 1])
    box_b = _boxes([[0, 0, 2, 2], [2, 2, 4, 4]], [0, 1])
    assert box_a == box_a
    assert box_a != box_b

    pose_a = PosePredictions(np.ones((2, 2, 2)), np.ones(2), np.ones(2), np.ones((2, 2), bool))
    pose_b = PosePredictions(np.ones((2, 2, 2)), np.ones(2), np.ones(2), np.ones((2, 2), bool))
    assert pose_a == pose_a
    assert pose_a != pose_b


def test_classification_constructor_validation():
    with pytest.raises(ValueError, match="prediction_format"):
        ClassificationPredictionFidelity(prediction_format="softmax")


def test_instance_constructor_validation_and_scalar_config():
    for bad_pos in [0, -0.1, 1.1, True, np.nan, np.inf]:
        with pytest.raises(ValueError, match="iou_threshold"):
            DetectionPredictionFidelity(iou_threshold=bad_pos)
        with pytest.raises(ValueError, match="oks_threshold"):
            PosePredictionFidelity(sigmas=[0.1], oks_threshold=bad_pos)

    for bad_conf in [-0.1, 1.1, True, np.nan, np.inf]:
        with pytest.raises(ValueError, match="confidence_threshold"):
            DetectionPredictionFidelity(confidence_threshold=bad_conf)
        with pytest.raises(ValueError, match="confidence_threshold"):
            PosePredictionFidelity(sigmas=[0.1], confidence_threshold=bad_conf)

    for bad_sigmas in [[], [0.0], [-0.1], [[0.1]], [np.nan], [True]]:
        with pytest.raises(ValueError, match="sigmas"):
            PosePredictionFidelity(sigmas=bad_sigmas)

    pose_metric = PosePredictionFidelity(sigmas=np.array([0.1, 0.2], dtype=np.float32))
    assert all(type(sigma) is float for sigma in pose_metric.sigmas)


def test_classification_validation_recovery_and_non_mutation():
    clf = ClassificationPredictionFidelity(prediction_format="logits", batch_size=1)
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    logits_copy = logits.copy()
    with pytest.raises(ValueError, match="same shape"):
        clf(np.ones((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        clf(np.array([[np.nan, 0.0]]), np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="iterable of pairs"):
        clf.evaluate_batches((logits, logits))
    with pytest.raises(ValueError, match="predictions"):
        clf.evaluate_batches(iter(()))
    assert clf(logits, logits)["top1_agreement"] == 1.0
    np.testing.assert_array_equal(logits, logits_copy)


def test_detection_validation_recovery_and_non_mutation():
    det = DetectionPredictionFidelity(confidence_threshold=0.5)
    box = _boxes([[0.0, 0.0, 2.0, 2.0]], [0], [0.9])
    box_snapshot = box.boxes.copy()
    with pytest.raises(ValueError, match="iterables"):
        det(box, box)
    with pytest.raises(ValueError, match="scores"):
        det([_boxes([[0, 0, 2, 2]], [0], [1.5])], [box])
    with pytest.raises(ValueError, match="class_ids"):
        det([_boxes([[0, 0, 2, 2]], [-1], [0.9])], [box])
    huge = _boxes([[0.0, 0.0, 1e200, 1e200]], [0], [0.9])
    with pytest.raises(ValueError, match="Box areas"):
        det([huge], [huge])
    with pytest.raises(ValueError, match="iterable of pairs"):
        det.evaluate_batches(([box], [box]))
    with pytest.raises(ValueError, match="at least one image"):
        det.evaluate_batches(iter(()))
    assert det([box], [box])["f1"] == 1.0
    np.testing.assert_array_equal(box.boxes, box_snapshot)


def test_pose_validation_recovery_and_confidence_filtering():
    pose = PosePredictionFidelity(sigmas=[0.1, 0.1], confidence_threshold=0.5)
    valid_pose = PosePredictions(
        np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        np.array([0.8]),
        np.array([25.0]),
        np.array([[True, True]]),
    )
    recon_pose = PosePredictions(np.array([[[0.0, 0.0], [1.0, 1.0]]]), np.array([0.8]))
    bool_ref = replace(valid_pose, keypoints=np.ones((1, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="real numerical"):
        pose([bool_ref], [recon_pose])
    with pytest.raises(ValueError, match="real numerical"):
        pose([valid_pose], [replace(recon_pose, keypoints=np.ones((1, 2, 2), dtype=bool))])
    with pytest.raises(ValueError, match="iterables"):
        pose(valid_pose, recon_pose)
    with pytest.raises(ValueError, match="Original areas"):
        pose([replace(valid_pose, areas=[0.0])], [recon_pose])
    with pytest.raises(ValueError, match="valid_joints"):
        pose([replace(valid_pose, valid_joints=[[1, 1]])], [recon_pose])
    low_conf = replace(recon_pose, scores=np.array([0.2]))
    filtered = pose([valid_pose], [low_conf])
    assert filtered["n_original"] == 1
    assert filtered["n_reconstructed"] == 0
    assert filtered["n_matched"] == 0
    assert pose([valid_pose], [recon_pose])["f1"] == 1.0
