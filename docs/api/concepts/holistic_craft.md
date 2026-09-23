# Holistic CRAFT

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/concepts/holistic_craft.py) |
📰 [CRAFT Paper](https://arxiv.org/pdf/2211.10154) |
📰 [Holistic Paper](https://arxiv.org/pdf/2306.07304)

Holistic CRAFT (Concept Recursive Activation FacTorization) is a variant of the CRAFT method designed to extract concepts from full activation maps rather than image patches.

This approach preserves the global spatial context and is particularly suitable for object detection models and other tasks where spatial structure across the entire image is important.

The crop-based approach works well for classification because images of classification datasets are typically dominated by a single, well-centred object: random crops are therefore likely to contain parts of the object of interest and carry relevant signal for concept extraction. In Object Detection, the scenes generally contain multiple objects of varying sizes, often occupying only a small fraction of the image. Random crops drawn from such images are mostly background; the target objects are absent or heavily under-represented in the resulting crop dataset, making the NMF factorization blind to the very patterns it should capture.

## Supported Object Detection Models

Holistic CRAFT works with various object detection architectures through specialized latent extractors provided by the `xplique-adapters` package:

**PyTorch (torchvision & ultralytics):**
- **RetinaNet** - `RetinanetExtractorBuilder`
- **Faster R-CNN** - `FasterRcnnExtractorBuilder`
- **FCOS** - `FcosExtractorBuilder`
- **SSD** - `SSDExtractorBuilder`
- **YOLO** (v11) - `YoloExtractorBuilder`
- **DETR** - `DetrExtractorBuilder`

**TensorFlow:**
- **RetinaNet** - `RetinaNetExtractorBuilder`

Each extractor handles the model-specific architecture to split it into the required g(.) and h(.) functions.

## Supported Classification Models

For standard classification models, Holistic CRAFT does not require a custom extractor per architecture. Instead, the built-in `LayeredModelExtractorBuilder` can split any layered model at a chosen intermediate layer:

**PyTorch:**
- Any `torch.nn.Module` — `LayeredModelExtractorBuilder` (from `xplique.concepts.torch.layered_model_latent_extractor`)

**TensorFlow:**
- Any `tf.keras.Model` — `LayeredModelExtractorBuilder` (from `xplique.concepts.tf.layered_model_latent_extractor`)

The builder takes the model and a layer index to define the split point. Everything before that layer becomes g(.), and everything after becomes h(.).

## Key Differences from Regular CRAFT

| Aspect | Regular CRAFT | Holistic CRAFT |
|--------|---------------|----------------|
| **Input** | Image patches/crops | Full activation maps |
| **Use Case** | Classification tasks | Object detection, Classification |
| **Spatial Context** | Local (patch-level) | Global (full image) |
| **Concepts** | Visual patterns in patches | Spatial activation patterns |
| **Performance** | Extracts many crops per image | Processes full feature maps directly |

## Workflow

Holistic CRAFT follows the same principle as CRAFT but operates on full images instead of
patches:

1. **Extract activations**: pass model inputs through the encoder $g$ to obtain intermediate
   activations.
2. **Factorize concepts**: factorize the activations to discover recurring concepts.
3. **Measure concept activation**: encode each image as concept activation maps in `coeffs_u`.
4. **Estimate concept importance**: attribute the task prediction to concept activations
   (by perturbing them with Sobol, or using a gradient-based method).
5. **Interpret concepts**: either visualize concept activation maps or perturb the input
   and attribute changes in concept activation scores to obtain localization maps.

!!!warning
    Activations must be non-negative to use the standard NMF. Ensure a ReLU
    or similar activation function is applied before the extraction layer.
    Third-party NMF implementations may not have this limitation
    (e.g., the Semi-NMF from the Overcomplete library).

The corresponding methods form a compact end-to-end sequence:

The example assumes that `images`, the task `operator`, `class_id`, and a configured
`partial_explainer` are already available.

```python
# 1. Fit concepts.
craft.fit(images)
# 2. Compute concept activation maps.
coeffs_u = craft.transform(images)
# 3. Estimate concept importance.
importances = craft.estimate_importance(images, operator, class_id)
# 4. Interpret concept activation maps.
craft.display_images_per_concept(images, coeffs_u=coeffs_u)
# Alternatively, compute concept localization maps with a PartialExplainer.
concept_maps = craft.attribute_concepts_to_inputs(images, partial_explainer)
```

Like regular CRAFT, Holistic CRAFT splits the model into $(g, h)$ such that
$f(x) = h(g(x))$. The encoder $g$ maps an input to latent activations, and $h$ maps those
activations to predictions. Concepts are extracted in this latent space.

This split is implemented through three abstractions:

- **`LatentData`**: A container that holds the intermediate activations produced by $g$. It abstracts away framework-specific tensor formats, providing a unified interface for reading (`get_activations`) and writing (`set_activations`) activations, with the necessary shape conversions (e.g., channel-first to channel-last).

- **`LatentExtractor`**: Wraps both $g$ (`input_to_latent_model`) and $h$ (`latent_to_logit_model`). It orchestrates the full forward pass, batching, device management, and output formatting. The `TorchLatentExtractor` and `TfLatentExtractor` subclasses provide framework-specific implementations.

- **`LatentExtractorBuilder`**: A factory that constructs a `LatentExtractor` for a specific model architecture. It handles all the architecture-specific wiring (defining how to split the model, which layer to extract from, and how to format outputs) so that the rest of the CRAFT pipeline remains model-agnostic.

## Concept Activation, Importance, and Localization

Holistic CRAFT exposes four related quantities with distinct meanings:

| Quantity | Definition | Question answered |
|---|---|---|
| **Concept activation map** | Spatial factorization coefficients $U_k(x)$, stored in `coeffs_u` | Where and how strongly is concept $k$ activated in latent space? |
| **Concept activation score** | A scalar reduction $s_k(x) = R(U_k(x))$ | How strongly is concept $k$ activated overall? |
| **Concept importance** | Attribution of the task prediction to concept activations | How much do the activations contribute to the task prediction? |
| **Concept localization map** | Attribution of a concept activation score to the input | Which input regions drive the activation score? |

For a CNN activation tensor `(N, H', W', C)`, factorization preserves the spatial axes and
produces `coeffs_u` with shape `(N, H', W', K)`. Thus, $U_k(x)$ is a concept activation map.
Spatial vision-transformer tokens can likewise form a concept activation map when the latent
extractor maps them back to their patch grid.
If the extracted representation has no spatial axes, `coeffs_u` has shape `(N, K)` and each
coefficient is already global; no spatial concept activation map is available to resize.

With the default spatial mean reducer, $s_k(x) = \operatorname{mean}_{i,j} U_k(x)_{i,j}$.
Resizing $U_k(x)$ therefore gives a CAM-style map for the *concept score*: the spatial map
underlying that score is displayed directly. Unlike Grad-CAM, this visualization does not
use gradients or class-prediction weights.

For perturbation-based methods, concept importance and localization look at different quantities:

```text
Concept importance (Sobol): perturb latent concept activations
    -> measure the task prediction -> attribute it to concepts

Concept localization (RISE/Sobol): perturb input regions
    -> measure concept activation score s_k(x) -> attribute it to input regions
```

Concept importance also supports gradient-based methods because Holistic CRAFT can decode
concept activations toward the task prediction. Input-to-concept localization instead follows
the fitted NumPy factorizer's `encode()` path, which is not differentiable, and therefore
requires a black-box attribution method.

### Interpretation Alternatives

Choose the representation that matches the question:

| Alternative | Use | Meaning |
|---|---|---|
| **Concept activation map visualization** | Pass `coeffs_u` to a display method | Resizes latent $U_k(x)$ as a CAM-style map for the concept score (with spatial mean reduction) |
| **Concept localization map** | Call `attribute_concepts_to_inputs()` and pass the result as `concept_maps` | Shows which input perturbations change $s_k(x)$ |

These alternatives can differ when latent maps are coarse or positions have large receptive
fields. Neither is a segmentation mask. Top-image ranking always uses `coeffs_u`, not the
magnitude of a concept localization map. Black-box localization is not inherently more
accurate than concept activation map visualization, and different black-box methods can
produce different maps.


## Example

### Basic Usage with Object Detection

```python
import xplique
from xplique.concepts import HolisticCraftTorch as Craft
from xplique_adapters.concepts.torch.latent_data_retinanet import RetinanetExtractorBuilder

# Build a latent extractor that splits the model into g(.) and h(.)
# This provides the input_to_latent (g) and latent_to_logit (h) functions
latent_extractor = RetinanetExtractorBuilder.build(
    model,
    device="cuda",
    nb_classes=91,
    extraction_location='resnet', # Choose 'resnet' or 'fpn'
    extraction_layer=-1  # Extract from last ResNet feature layer
)

# Create Holistic CRAFT instance
craft = Craft(
    latent_extractor=latent_extractor,
    number_of_concepts=10,
    device="cuda"
)

# Fit CRAFT on input images to discover concepts
craft.fit(input_images, class_id=class_id)

# Display concept activation maps overlaid on images
craft.display_images_per_concept(images=input_images[:5])

# Display top 3 images for each concept ranked by concept activation score
craft.display_top_images_per_concept(images=input_images, topk=3)

# Estimate concept importance on the 20 first images using Gradient×Input method
# (GradientxInput is the default method)
importances_gi = craft.estimate_importance(
    images=input_images[:20],
    operator=xplique.Tasks.OBJECT_DETECTION,
    class_id=class_id,
    confidence=0.8
)

# Estimate concept importance on the 20 first images using Sobol method
importances_sobol = craft.estimate_importance(
    images=input_images[:20],
    operator=xplique.Tasks.OBJECT_DETECTION,
    class_id=class_id,
    confidence=0.8,
    # Use Sobol method & its arguments
    method="sobol",
    grid_size=4,
    nb_design=8,
    perturbation_function="amplitude",
)

```

### Using Different Attribution Methods to Compute Concept Importance

Holistic CRAFT supports various attribution methods for concept importance estimation:

```python
import xplique
from xplique.concepts import PartialExplainer
from xplique.attributions import VarGrad

# Use VarGrad for robust importance estimation
vargrad_explainer = PartialExplainer(
    explainer_class=VarGrad,
    operator=xplique.Tasks.OBJECT_DETECTION,
    nb_samples=20,
    noise=0.15
)

# Compute VarGrad explanation for each concept
explanation_vargrad = craft.compute_explanation_per_concept(
    partial_explainer=vargrad_explainer,
    images=input_images,
    class_id=class_id,
    confidence=0.3,
)

# Reduce the spatial dimensions to compute concept importance
importances_vargrad = craft.reduce_to_importance(
    explanation=explanation_vargrad,
)
```

## Localizing Concepts with Black-Box Attribution

`attribute_concepts_to_inputs()` exposes the fitted encoder and factorizer as a callable that
returns one concept activation score per learned concept. It creates one-hot concept targets
and applies a compatible black-box explainer to each requested concept. No task `operator` is
needed because the targets select concept activation scores directly.

For an input batch with shape `(N, H, W, C)`, the returned maps have shape
`(N, H, W, number_of_concepts)`. Channel `k` always corresponds to concept `k`. When only a
subset is requested, uncomputed channels are filled entirely with `NaN` so that they cannot
be confused with valid zero attributions.

TensorFlow callers pass channel-last images. PyTorch callers pass their native channel-first
`(N, C, H, W)` images to the same high-level method; Xplique handles the layout conversion
for the wrapped localizer.

### Sobol Example

```python
from xplique.attributions import SobolAttributionMethod
from xplique.concepts import PartialExplainer

sobol = PartialExplainer(
    SobolAttributionMethod,
    grid_size=8,
    nb_design=32,
    perturbation_function="inpainting",
)

# `model_images` are preprocessed inputs; `display_images` are matching display images.
concept_maps = craft.attribute_concepts_to_inputs(
    model_images,
    partial_explainer=sobol,
    concept_ids=[0, 3, 7],
    concept_reducer="mean",
)

craft.display_images_per_concept(
    display_images,
    concept_maps=concept_maps,
    order=[0, 3, 7],
)
```

`concept_reducer="mean"` reduces every $U_k(x)$ to the concept activation score attributed to
the input. It is the default and matches the reduction used to rank representative images.
The localizer preserves signed scores; use a custom callable reducer only when another scalar
definition is intended.

`nb_design` must be a nonzero power of two. Keep Sobol's default `nb_channels=1`: localization
computes a separate single-channel map for each selected concept. Sobol-based concept
importance is a different operation: it perturbs latent concept activations and attributes
changes in the task prediction to concepts, rather than perturbing inputs to measure $s_k(x)$.

RISE is an alternative black-box configuration; pass `rise` as `partial_explainer` to the same
method:

```python
from xplique.attributions import Rise

rise = PartialExplainer(
    Rise, nb_samples=2000, grid_size=7, preservation_probability=0.5, mask_value=0.0
)
```

When displaying top images with a concept localization map, `coeffs_u` still determines the
ranking and `concept_maps` determines only the overlay:

```python
coeffs_u = craft.transform(model_images)
craft.display_top_images_per_concept(
    display_images,
    coeffs_u=coeffs_u,
    concept_maps=concept_maps,
    order=[0, 3, 7],
    topk=3,
)
```

If `order` is omitted, display methods show only the concept channels that were computed.
Explicitly requesting an uncomputed `NaN` channel raises a `ValueError`. Signed localization
maps are displayed by absolute magnitude; attribution direction is not represented by the
current renderer.

!!!warning "Computational cost"
    Cost scales with concepts x images x perturbations because localization evaluates the
    encoder and `factorizer.encode()` for each perturbation and selected concept. Rank concepts
    by importance first, then localize a small subset. Reduce images, samples or designs, and
    grid resolution for exploratory runs.

!!!warning "Model and display inputs"
    `model_images` must be model-ready, preprocessed inputs; perturbations operate in that
    space. For example, zero after ImageNet normalization represents the dataset mean, not a
    raw black pixel. `display_images` may instead contain human-readable RGB images, but must
    have the same order and spatial correspondence as `model_images`.

!!!warning "Factorizer compatibility"
    Localization evaluates the fitted factorizer on unseen, perturbed activations. The
    factorizer must therefore support out-of-sample `encode()`. Activations should not be
    clipped merely to satisfy a factorizer because that would change the function being
    explained.

!!!warning "Black-box scope"
    Input-to-concept localization supports black-box attribution methods only. Do not use
    Integrated Gradients for localization: the NumPy `factorizer.encode()` path is not
    differentiable. This does not prevent gradient-based concept importance, which follows
    the separate concept-activation-to-task-prediction path.

## Using a Different NMF Factorizer

By default, the standard Sklearn NMF is used to factorize the concepts.
But other types of factorizers are supported, such as the ones provided
by the [Overcomplete](https://github.com/KempnerInstitute/overcomplete) project.

```python
from overcomplete.optimization import SemiNMF
from xplique.concepts.torch.factorizer import OvercompleteFactorizer

nb_concepts=10

# Create a SemiNMF factorizer which allows negative activations
factorizer = OvercompleteFactorizer(
    optimizer_class=SemiNMF,
    nb_concepts=nb_concepts,
    device=device
)

# Setup Craft to use this factorizer
craft = Craft(
    latent_extractor=latent_extractor,
    number_of_concepts=nb_concepts,
    device=device,
    factorizer=factorizer,
)

craft.fit(input_images)
```

## Implementing Your Own Latent Extractor

If you're working with a model architecture that isn't supported out-of-the-box, you can implement your own latent extractor by following these steps:

### 1. Create a Custom LatentData Class

First, create a class that stores the intermediate activations from your model:

```python
from xplique.concepts.latent_extractor import LatentData
import torch

class CustomLatentData(LatentData):
    def __init__(self, fpn_outs: list, extraction_layer: int = 0):
        super().__init__()
        self.fpn_outs = fpn_outs
        self.extraction_layer = extraction_layer

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False):
        """Extract activations from the specified layer."""
        activations = self.fpn_outs[self.extraction_layer]

        if not keep_gradients:
            activations = activations.detach()

        # Convert from (N, C, H, W) to (N, H, W, C) for Xplique
        if len(activations.shape) == 4:
            activations = activations.permute(0, 2, 3, 1)

        if as_numpy:
            activations = activations.cpu().numpy()

        return activations

    def set_activations(self, values: torch.Tensor) -> None:
        """Set activations back into the latent data structure."""
        # Convert from (N, H, W, C) to (N, C, H, W)
        if len(values.shape) == 4:
            values = values.permute(0, 3, 1, 2)
        self.fpn_outs[self.extraction_layer] = values

    def to(self, device: torch.device) -> 'CustomLatentData':
        """Move latent data to specified device."""
        self.fpn_outs = [fpn_out.to(device) for fpn_out in self.fpn_outs]
        return CustomLatentData(self.fpn_outs, self.extraction_layer)
```

### 2. Create a Custom ExtractorBuilder

Next, implement a builder that splits your model into g(.) and h(.) functions:

```python
import types
from xplique.concepts.latent_extractor import LatentExtractorBuilder
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

class CustomExtractorBuilder(LatentExtractorBuilder):
    @classmethod
    def build(
        cls,
        model,
        device: str = 'cuda',
        extraction_layer: int = -1,
        batch_size: int = 1
    ) -> TorchLatentExtractor:

        # Define g(.) function: input → latent activations
        def g(self, x):
            # Example: extract from backbone/feature pyramid
            fpn_outs = self.backbone(x)
            return CustomLatentData(
                fpn_outs=list(fpn_outs),
                extraction_layer=latent_extractor.extraction_layer
            )

        # Define h(.) function: latent activations → predictions
        def h(self, latent_data: CustomLatentData):
            fpn_outs = latent_data.fpn_outs
            outputs = self.head(fpn_outs)
            return outputs

        # Bind g and h methods to the model
        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        # Create output formatter (converts raw predictions to MultiBoxTensor)
        output_formatter = CustomBoxFormatter()

        # Build the latent extractor
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=CustomLatentData,
            output_formatter=output_formatter,
            batch_size=batch_size,
            device=device
        )

        # Store extraction layer for later use
        latent_extractor.extraction_layer = extraction_layer
        return latent_extractor
```

### 3. Use Your Custom Extractor with CRAFT

Once you have your custom extractor, you can use it just like the built-in ones:

```python
from xplique.concepts import HolisticCraftTorch as Craft

# Build your custom latent extractor
latent_extractor = CustomExtractorBuilder.build(
    model,
    device="cuda",
    extraction_layer=-1,
    batch_size=16
)

# Use it with CRAFT
craft = Craft(
    latent_extractor=latent_extractor,
    number_of_concepts=10,
    device="cuda"
)

# Fit and visualize concepts
craft.fit(input_images)
craft.display_images_per_concept(input_images[:5])
```

### Key Points

- **g(.) function**: Maps input images to intermediate activations at a chosen layer
- **h(.) function**: Maps latent activations back to final predictions
- **LatentData**: Handles activation extraction with proper shape conversions (PyTorch uses channel-first, Xplique expects channel-last)
- **Output formatter**: Converts model predictions to `MultiBoxTensor` format for compatibility with Xplique


## API Reference

{{xplique.concepts.holistic_craft.HolisticCraft}}

{{xplique.concepts.holistic_craft.PartialExplainer}}

## References

[^1]: [CRAFT: Concept Recursive Activation FacTorization for Explainability (2023).](https://arxiv.org/pdf/2211.10154.pdf)

[^2]: [A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation (2023).](https://arxiv.org/pdf/2306.07304.pdf)
