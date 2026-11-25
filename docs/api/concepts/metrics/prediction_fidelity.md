# Prediction Fidelity After Reconstruction

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/prediction_fidelity.py)

A good activation reconstruction can still change a model's predictions. Compare
the original model's predictions to those from the **same downstream head** after
replacing the held-out activations with their factorizer reconstruction. The
scorers on this page require no labels and don't run the model.

## Preparing paired predictions

1. Use held-out inputs and put the model and factorizer in their evaluation modes.
   Extract activations once per batch. Flatten channels-last activations `(..., F)`
   to `(-1, F)` for factorizers that expect a matrix.
2. Reconstruct with `factorizer.decode(factorizer.encode(X))`. Using `U @ D`
   instead can omit an affine decoder's bias. Restore activation shape, layout,
   dtype, and device before calling the downstream head. For a split with multiple
   inputs to the head, replace only the features being studied; keep skip connections
   and other feature maps unchanged and don't mutate the original latent container.
3. Run both branches through the same head and postprocessing in the same image
   order, then convert the decoded outputs to NumPy. Use the same resize, coordinate
   restoration, class vocabulary, and filtering rules on both sides.

Each scorer is configured once and accepts data per call. Use `evaluate_batches`
to pool one-pass prediction pairs without keeping predictions for the whole dataset.
Do not average batch reports: their denominators differ.

## Classification

Pass `(N, C)` single-label logits or normalized class probabilities. Top-1 agreement
checks decisions. Mean KL in nats compares the original distribution to the
reconstructed one; it is directional and can be infinite when reconstructed
probability is zero for a class with positive original probability. Use logits if
available to avoid rounding positive probabilities to zero.

The toy factorizer below keeps a high-variance direction and reconstructs the second
coordinate from an affine bias. Latent $R^2$ is high, but the head uses the second
coordinate and changes one of its predictions.

```python
import numpy as np
from xplique.metrics.concepts import ClassificationPredictionFidelity, ReconstructionR2

def encode(activations):
    return activations[:, :1]

def decode(codes):
    return np.concatenate((codes, np.full_like(codes, -0.01)), axis=1)

activations = np.array([[1.0, 0.01], [100.0, -0.02]])
reconstructed = decode(encode(activations))
head_weights = np.array([[0.0, 0.0], [1.0, -1.0]])
metric = ClassificationPredictionFidelity(prediction_format="logits")
print(ReconstructionR2()(activations, reconstructed))
print(metric(activations @ head_weights, reconstructed @ head_weights))
```

## Object detection

Pass one `BoxPredictions` per image. Each contains unpadded `xyxy` boxes, integer
class IDs, and the confidence used for filtering. For Xplique formatted detections
(`TfMultiBoxTensor` or `TorchMultiBoxTensor`), take `.boxes()`, `.scores()` and the
argmax of `.probas()` for the class IDs, then convert to NumPy. For a padded tensor,
strip padding before scoring rather than treating it as a detection.

The scorer matches within each image and class. It maximizes the number of matches
at or above the IoU threshold first, then total IoU. Precision uses the reconstructed
instance count; recall uses the original count. Mean matched IoU is conditional on
matching. Missing or extra instances are reflected in F1, not the conditional mean.

```python
import numpy as np
from xplique.metrics.concepts import BoxPredictions, DetectionPredictionFidelity

original = [BoxPredictions(np.array([[0, 0, 10, 10]]), np.array([0]), np.array([0.9]))]
reconstructed = [
    BoxPredictions(
        np.array([[0, 0, 9, 10], [50, 50, 60, 60]]), np.array([0, 0]), np.array([0.8, 0.7])
    )
]
print(DetectionPredictionFidelity(confidence_threshold=0.5)(original, reconstructed))
```

## Pose estimation

Pass one `PosePredictions` per image, for a single skeleton shared by both branches.
Supply positive original instance areas in squared coordinate units and a boolean
validity mask for original joints. Reconstructed joint confidence does not change the
scored joint set. An original pose that passes the instance-confidence filter must
have at least one valid joint. Masked original coordinates may be missing.

Matching uses thresholded Object Keypoint Similarity (OKS), with each squared
distance divided by $2A(2\sigma_j)^2$ inside the negative exponential. The caller
supplies sigmas in skeleton order; a custom skeleton must not silently inherit COCO
person sigmas. For YOLO-NAS Pose, apply the same pose postprocessing callback and
original-image coordinate restoration to both outputs. Split decoded `poses` into
`[..., :2]` coordinates and joint confidence (if present), take instance `scores`
separately, and explicitly choose the reference validity rule and area definition.
The example uses an area and validity mask already resolved at that boundary.

```python
import numpy as np
from xplique.metrics.concepts import PosePredictionFidelity, PosePredictions

coords = np.array([[[4, 4], [8, 8]]])
original = [PosePredictions(coords, np.array([0.9]), np.array([100]), np.array([[True, True]]))]
reconstructed = [PosePredictions(coords + np.array([1, 0]), np.array([0.8]))]
print(PosePredictionFidelity(sigmas=[0.1, 0.1])(original, reconstructed))
```

For both detection and pose, an image with no instances on either side adds no
instances to the pooled counts. A ratio with no denominator is `NaN`. A missing
prediction on just one side still lowers F1. The matched scores are **not** AP.

## Semantic segmentation

Pass integer label maps on the same grid with class IDs in `[0, n_classes)`. There
is no implicit resize or `argmax`. A boolean mask can exclude unlabeled, padded,
or otherwise ignored pixels from both sides. Pixel agreement and per-class IoUs are
computed from a pooled confusion matrix. Macro IoU excludes classes absent from
both sides; it counts classes appearing on only one side as zero IoU.

```python
import numpy as np
from xplique.metrics.concepts import SemanticSegmentationPredictionFidelity

original = np.array([[0, 0], [1, -1]])
reconstructed = np.array([[0, 1], [1, 255]])
metric = SemanticSegmentationPredictionFidelity(n_classes=2)
print(metric(original, reconstructed, valid_mask=original != -1))
```

An evaluation with only ignored pixels has undefined agreement and IoU. Instance
and panoptic segmentation are not covered by this scorer.

## Separately report task performance

Prediction agreement doesn't tell you whether either branch is correct. When
annotations are available, score both branches against the **same annotations** and
report the original score, the reconstructed score, and `reconstructed − original`.

| Task | Ground-truth report |
| --- | --- |
| Classification | Accuracy and cross-entropy against labels. |
| Object detection | Dataset box AP from an existing evaluator such as `COCOeval(iouType="bbox")`, with the same image and category IDs and evaluator settings. |
| Pose estimation | Dataset keypoint AP from `COCOeval(iouType="keypoints")` or a dataset-specific evaluator, with fixed skeleton, sigmas, coordinate conventions, and annotations. |
| Semantic segmentation | Dataset mIoU against annotated masks, with the same label map, ignore mask, and background rule. |

For classification, an optional [SAEBench-style](https://arxiv.org/abs/2503.09532)
loss recovery score is $(L_\mathrm{ablated}-L_\mathrm{reconstructed}) /
(L_\mathrm{ablated}-L_\mathrm{original})$. Report all three losses and the
explicit latent ablation. Leave recovery undefined unless the ablation worsens
the original loss. Don't clamp a valid score to `[0, 1]` or apply this
normalization across tasks by default.

## API Reference

{{xplique.metrics.concepts.prediction_fidelity.ClassificationPredictionFidelity}}

{{xplique.metrics.concepts.prediction_fidelity.DetectionPredictionFidelity}}

{{xplique.metrics.concepts.prediction_fidelity.PosePredictionFidelity}}

{{xplique.metrics.concepts.prediction_fidelity.SemanticSegmentationPredictionFidelity}}

{{xplique.metrics.concepts.prediction_fidelity.BoxPredictions}}

{{xplique.metrics.concepts.prediction_fidelity.PosePredictions}}
