# Object detection adapters

[`xplique-adapters`](https://github.com/deel-ai/xplique-adapters) is the optional,
model-specific integration layer for Xplique. Release
[0.1.0](https://github.com/deel-ai/xplique-adapters/releases/tag/0.1.0) provides
prediction formatters, detection wrappers, and latent extractors for pixel attribution
and [Holistic CRAFT](../concepts/holistic_craft.md).

## Installation and supported models

Install the extra for your model ecosystem:

```bash
pip install "xplique-adapters[torchvision]"
pip install "xplique-adapters[yolo]"
pip install "xplique-adapters[detr]"
pip install "xplique-adapters[retinanet]"
```

These are independent installation choices. Multiple extras can be combined, for example
`pip install "xplique-adapters[yolo,torchvision]"`. The package name uses a hyphen;
Python imports use `xplique_adapters`.

| Model ecosystem | Extra | Detection wrapper | Holistic CRAFT builder |
|---|---|---|---|
| torchvision RetinaNet | `torchvision` | `TorchvisionBoxesModelWrapper` | `RetinaNetExtractorBuilder` |
| torchvision Faster R-CNN | `torchvision` | `TorchvisionBoxesModelWrapper` | `FasterRcnnExtractorBuilder` |
| torchvision FCOS | `torchvision` | `TorchvisionBoxesModelWrapper` | `FcosExtractorBuilder` |
| torchvision SSD (MobileNetV3 extractor) | `torchvision` | `TorchvisionBoxesModelWrapper` | `SSDExtractorBuilder` |
| Ultralytics YOLO 11 / YOLO 26 | `yolo` | Results or raw-output wrappers; see below | `YoloExtractorBuilder` |
| HuggingFace DETR | `detr` | `DetrBoxesModelWrapper` | `DetrExtractorBuilder` |
| Facebook-reference DETR | `torchvision` | `DetrBoxesModelWrapper` | `DetrExtractorBuilder` |
| KerasCV RetinaNet | `retinanet` | `RetinaNetBoxesModelWrapper` | `RetinaNetExtractorBuilder` (TensorFlow) |

Detection wrappers are exported from `xplique_adapters.object_detection.torch` or
`xplique_adapters.object_detection.tf`. Builders are imported from architecture modules
such as `xplique_adapters.concepts.torch.latent_data_retinanet`; they are not exported
from the concepts package root.

Adapters 0.1.0 require Xplique >=2.0.0 and Python 3.10-3.13. The `torch` extra requires
PyTorch >=2.6,<2.11; `detr` requires >=2.7,<2.11 and Transformers >=5,<6. The `yolo`
extra supports Ultralytics >=8.4.120,<8.5. TensorFlow support targets >=2.18,<2.21
and KerasCV >=0.9,<0.10. Use mutually compatible PyTorch and torchvision versions.
Facebook-reference DETR is loaded separately by the caller, typically through Torch Hub.

## Choose an integration path

| Goal | Components |
|---|---|
| Inspect or plot detections | Formatter or detection wrapper |
| Explain input pixels for selected detections | Detection wrapper + task operator + attribution method; PyTorch also uses `TorchWrapper` |
| Discover concepts and measure their importance | Latent extractor builder + `HolisticCraftTorch` or `HolisticCraftTf` |
| Localize a learned concept in the input | Fitted Holistic CRAFT + black-box `PartialExplainer` |

A **formatter** converts a detector's output to the Xplique box representation. A
**detection wrapper** calls the detector and formats its predictions. A **latent
extractor** additionally splits the detector into an encoder $g$ and a decoder $h$,
retaining companion feature maps, skip connections, and metadata needed to decode
modified activations. A detection wrapper alone does not provide this latent split.

### Output format and tensor mode

The default output is a list containing one `MultiBoxTensor` per image. Each element
has shape `(num_boxes, 5 + nb_classes)` and stores:

```text
[x_min, y_min, x_max, y_max, detection_confidence, class_scores...]
```

This supports varying detection counts and helpers such as `.filter(class_id=...,
confidence=...)`, `.boxes()`, `.scores()`, and `.probas()`.

The shared attribution operator needs a batched tensor. After selecting reference
boxes in list mode, set `adapter.output_as_tensor = True` (equivalently,
`adapter.output_as_list = False`). Tensor mode produces
`(N, max_num_boxes, 5 + nb_classes)` with all-zero padding. The object detection
operator ignores padding and returns a zero score for an image with no predictions
or no references. Keep reference boxes fixed throughout an attribution call.

### Prediction stage and method selection

| Path | Output stage | Guidance |
|---|---|---|
| `TorchvisionBoxesModelWrapper` | Standard postprocessed detections, usually after thresholding/NMS | Suitable for black-box attribution; gradients depend on the retained prediction path |
| `YoloResultBoxesModelWrapper(yolo)` | Public Ultralytics `list[Results]` | Inference and black-box attribution; pass the public YOLO object, not `yolo.model` |
| `Yolo11RawBoxesModelWrapper(yolo.model)` | One-to-many decoded predictions before Results/NMS | Use for a differentiable YOLO 11 path |
| `Yolo26RawBoxesModelWrapper(yolo.model, nb_classes=...)` | One-to-one decoded top-k predictions before Results | Requires `yolo.model.end2end = True` |
| `DetrBoxesModelWrapper` | Query predictions decoded from logits and normalized boxes | No conventional anchor/NMS stage; supports Facebook and HuggingFace outputs |
| `RetinaNetBoxesModelWrapper` (TensorFlow) | Decoded predictions | `prediction_mode="raw"` calls `decode_predictions`; `"decoded"` accepts already-decoded outputs |

Black-box methods can explain the complete postprocessing pipeline. Thresholding and
NMS affect which boxes remain available for matching; pre-NMS predictions provide a
larger candidate pool. Gradient methods need outputs connected to the inputs. Wrapping
an inference-only path cannot restore lost gradients. Discrete class labels can be
encoded as one-hot vectors for matching, but do not provide soft class gradients.

YOLO 26's end-to-end flag defaults to false from Ultralytics 8.4.142. The raw one-to-one
wrapper requires it to be enabled. The latent `YoloExtractorBuilder` instead selects
`YoloExtractorMode.ONE_TO_MANY` or `ONE_TO_ONE` explicitly on a copy of the model.

### Coordinates, confidence, and class IDs

Predictions and references must use the same coordinate system and class indices.
Torchvision, DETR, and YOLO adapters output absolute XYXY coordinates; the TensorFlow
RetinaNet adapter outputs normalized XYXY. Public DETR and TensorFlow RetinaNet
`image_size` parameters use `(height, width)`.

The confidence column is model-specific: torchvision uses its reported score, and
DETR uses the largest foreground class probability. It need not be a separate
objectness probability. Torchvision labels are one-hot encoded; DETR keeps foreground
softmax scores and excludes the final no-object class.

For COCO torchvision models, use the model's category metadata and `nb_classes=91`;
the label space includes background and unused slots. Ultralytics COCO models usually
use 80 classes with different indices. Do not transfer class IDs between ecosystems.
For normalized boxes, pass the corresponding `box_type` when plotting.

## PyTorch pixel attribution

The following example assumes `model_images` is a float32 PyTorch batch shaped
`(N, C, H, W)` in `[0, 1]`. Torchvision performs its own internal normalization.
Image loading, fixed-reference selection, group attribution, and visualization follow
the same pattern: inspect predictions in list mode, freeze references, then switch to
tensor mode for attribution.

```python
import numpy as np
import torch
from torchvision.models.detection import fcos_resnet50_fpn

import xplique
from xplique.attributions import Occlusion
from xplique.wrappers import TorchWrapper
from xplique_adapters.object_detection.torch import TorchvisionBoxesModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
model = fcos_resnet50_fpn(weights="DEFAULT").eval().to(device)
adapter = TorchvisionBoxesModelWrapper(model, nb_classes=91)
with torch.no_grad():
    boxes = adapter(model_images[:1].to(device))[0].filter(class_id=1, confidence=0.3)
if len(boxes) == 0:
    raise ValueError("No person detected; choose another image or confidence threshold.")
targets = boxes[:1].detach().cpu().numpy()[None, ...]

adapter.output_as_tensor = True
wrapped_model = TorchWrapper(
    adapter, device=device, is_channel_first=True, requires_grad=False
)
inputs = np.moveaxis(model_images[:1].detach().cpu().numpy(), 1, -1)
explainer = Occlusion(
    wrapped_model, operator=xplique.Tasks.OBJECT_DETECTION,
    patch_size=64, patch_stride=64, batch_size=1,
)
explanations = explainer(inputs, targets)
```

`TorchWrapper` receives channel-last inputs and converts them to the detector's
channel-first layout. `requires_grad=False` is appropriate for this black-box example;
gradient methods require `True` and a differentiable detector path. Native
`HolisticCraftTorch` calls instead accept channel-first images and manage the decoder
bridge internally.

## Concept explanations

For RetinaNet, the extractor selects a ResNet or FPN feature level and preserves the
other features and image metadata needed by the decoder. Importance describes changes
at that selected level while the companion context is retained.

`craft.fit(images, class_id=...)` stores the class ID as metadata; it does not filter
the fitting dataset or activation locations. For class-focused discovery, curate the
input images. During importance estimation, `class_id` and `confidence` select reference
detections; images without matches receive zero explanations. Concept localization
instead explains a learned concept activation score, without an object detection operator.

See the [companion README](https://github.com/deel-ai/xplique-adapters/tree/0.1.0#readme)
for model-specific arguments and its
[examples](https://github.com/deel-ai/xplique-adapters/tree/0.1.0/examples) for additional
architectures. Model-specific adapters retain their upstream licenses; details are in
the companion repository's [license documentation](https://github.com/deel-ai/xplique-adapters/blob/0.1.0/LICENSE).
