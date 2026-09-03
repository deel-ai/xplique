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

Holistic CRAFT follows the same core principle as CRAFT but operates on full images instead of patches:

1. **Extract Activations**: Pass input images through the model's encoder (g) to obtain spatial activation maps from an intermediate layer
2. **Factorize Concepts**: Apply Non-negative Matrix Factorization (NMF) to these activation maps to discover recurring spatial patterns (concepts)

!!!warning
    Activations must be non-negative to use the standard NMF. Ensure a ReLU
    or similar activation function is applied before the extraction layer.
    Third-party NMF implementations may not have this limitation
    (e.g., the Semi-NMF from the Overcomplete library).

3. **Measure Concept Activation**: Use concept coefficients to measure how strongly each concept is present and rank representative images
4. **Estimate Concept Importance**: Attribute task predictions to concept coefficients to measure how much each concept contributes to the model output
5. **Visualize Coefficients**: Resize latent coefficient maps and overlay them on input images
6. **Localize Concepts**: Optionally use a black-box attribution method to identify which input regions drive a selected concept score

Like regular CRAFT, Holistic CRAFT requires splitting the model into two parts: $(g, h)$ such that $f(x) = (g \cdot h)(x)$. The model $g$ maps input to latent space (activation maps), and $h$ maps latent space to predictions. Concepts are extracted from these activation maps in latent space.

This split is implemented through three abstractions:

- **`LatentData`**: A container that holds the intermediate activations produced by $g$. It abstracts away framework-specific tensor formats, providing a unified interface for reading (`get_activations`) and writing (`set_activations`) activations, with the necessary shape conversions (e.g., channel-first to channel-last).

- **`LatentExtractor`**: Wraps both $g$ (`input_to_latent_model`) and $h$ (`latent_to_logit_model`). It orchestrates the full forward pass, batching, device management, and output formatting. The `TorchLatentExtractor` and `TfLatentExtractor` subclasses provide framework-specific implementations.

- **`LatentExtractorBuilder`**: A factory that constructs a `LatentExtractor` for a specific model architecture. It handles all the architecture-specific wiring (defining how to split the model, which layer to extract from, and how to format outputs) so that the rest of the CRAFT pipeline remains model-agnostic.

## Concept Activation, Importance, and Localization

Holistic CRAFT exposes three related quantities that answer different questions:

| Quantity | Definition | Question answered |
|---|---|---|
| **Concept activation** | The spatial coefficient map $U_k(x)$, or a reduction of it | How strongly is concept $k$ present? |
| **Concept importance** | Attribution of the final task prediction to concept $k$ | How much does concept $k$ contribute to the prediction? |
| **Concept localization** | Attribution of the concept score $s_k(x) = R(U_k(x))$ to the input | Which input regions drive the concept score? |

Concept importance and concept localization follow opposite attribution directions:

```text
concept coefficients -> final prediction -> concept importance

input image -> concept scores -> input attribution -> concept localization
```

A coefficient heatmap is a latent-space map resized to the input resolution. A localization
map is an input-space attribution produced by perturbing the input and observing changes in a
selected concept score. The maps can therefore differ, especially when the latent
representation is coarse or its positions have large receptive fields.

Top-image ranking always uses concept coefficients because they represent concept presence.
Attribution-map magnitude is not a replacement for concept activation.

!!!note
    Coefficient and localization maps are explanations, not segmentation masks. Black-box
    localization is not automatically more correct than coefficient visualization; it answers
    the more specific question of which input perturbations change a selected concept score.


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

# Display discovered concepts as heatmaps overlaid on images
craft.display_images_per_concept(images=input_images[:5])

# Display top 3 images for each concept ranked by activation
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

### Using Different Attribution Methods to Compute the Concept Importances

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

# Reduce the spatial dimension of the explanation
# to compute the final concepts importances
importances_vargrad = craft.reduce_to_importance(
    explanation=explanation_vargrad,
)
```

## Localizing Concepts with Black-Box Attribution

`ConceptLocalizer` exposes the fitted encoder and factorizer as a callable that returns one
scalar score per learned concept. `compute_concept_attributions()` builds this callable,
constructs one-hot concept targets, and applies a compatible black-box explainer to each
requested concept.

For an input batch with shape `(N, H, W, C)`, the returned maps have shape
`(N, H, W, number_of_concepts)`. Channel `k` always corresponds to concept `k`. When only a
subset is requested, uncomputed channels are filled entirely with `NaN` so that they cannot
be confused with valid zero attributions.

TensorFlow callers pass channel-last images. PyTorch callers pass their native channel-first
`(N, C, H, W)` images to the same high-level method; Xplique handles the layout conversion
for the wrapped localizer.

### Localizing Concepts with RISE

```python
from xplique.attributions import Rise
from xplique.concepts import PartialExplainer

rise = PartialExplainer(
    Rise,
    nb_samples=2000,
    grid_size=7,
    preservation_probability=0.5,
    mask_value=0.0,
)

rise_maps = craft.compute_concept_attributions(
    images,
    partial_explainer=rise,
    concept_ids=[0, 3, 7],
    concept_reducer="mean",
)

craft.display_images_per_concept(
    display_images,
    concept_maps=rise_maps,
    order=[0, 3, 7],
)
```

`concept_reducer="mean"` reduces every spatial coefficient map to the scalar score that RISE
attributes to the input. It is the default and is consistent with the mean coefficient
activation used to rank representative images. The localizer preserves signed scores; a
custom callable reducer can be used when concept magnitude is intended instead.

Do not pass a task `operator` to `PartialExplainer` for concept localization. One-hot targets
are created internally to select concept scores directly.

### Localizing Concepts with Sobol

The same API works with `SobolAttributionMethod`:

```python
from xplique.attributions import SobolAttributionMethod

sobol = PartialExplainer(
    SobolAttributionMethod,
    grid_size=8,
    nb_design=32,
    perturbation_function="inpainting",
)

sobol_maps = craft.compute_concept_attributions(
    images,
    partial_explainer=sobol,
    concept_ids=[0, 3, 7],
)

craft.display_images_per_concept(
    display_images,
    concept_maps=sobol_maps,
    order=[0, 3, 7],
)
```

`nb_design` must be a nonzero power of two. Keep Sobol's default `nb_channels=1`: concept
localization computes a separate single-channel attribution map for each selected concept.
This differs from Sobol-based concept importance, which attributes the final task prediction
to concept coefficients.

### Coefficient Maps and Localization Maps

Use `coeffs_u` to display latent coefficient maps:

```python
craft.display_images_per_concept(
    display_images,
    coeffs_u=coefficients,
    order=selected_concepts,
)
```

Use `concept_maps` to display input-space localization maps:

```python
craft.display_images_per_concept(
    display_images,
    concept_maps=rise_maps,
    order=selected_concepts,
)
```

When displaying the top images, coefficients still determine the ranking and localization
maps only determine the overlay:

```python
craft.display_top_images_per_concept(
    display_images,
    coeffs_u=coefficients,
    concept_maps=rise_maps,
    order=selected_concepts,
    topk=3,
)
```

If `order` is omitted, display methods show only the concept channels that were computed.
Explicitly requesting an uncomputed `NaN` channel raises a `ValueError`. Signed localization
maps are displayed by absolute magnitude; attribution direction is not represented by the
current renderer.

!!!warning "Computational cost"
    Black-box localization evaluates the encoder and `factorizer.encode()` for many perturbed
    inputs and runs a separate attribution pass for every selected concept. Estimate concept
    importance first, then localize only a small number of important concepts. Reduce the
    number of images, RISE samples, Sobol designs, or grid resolution for exploratory runs.

!!!warning "Perturbations use preprocessed inputs"
    Perturbation parameters operate in the model's preprocessed input space. For example,
    `mask_value=0.0` is neutral or black only if zero has that meaning after preprocessing.
    With standard ImageNet normalization, zero generally corresponds to the dataset mean
    rather than a raw black pixel. Sobol inpainting and blurring must be interpreted relative
    to the same model-ready representation.

!!!warning "Factorizer compatibility"
    Localization evaluates the fitted factorizer on unseen, perturbed activations. The
    factorizer must therefore support out-of-sample `encode()`. Activations should not be
    clipped merely to satisfy a factorizer because that would change the function being
    explained.

!!!warning "Black-box scope"
    Input-to-concept localization currently supports black-box attribution methods only.
    Ordinary `factorizer.encode()` is not guaranteed to be differentiable, so white-box
    explainers such as Gradient Input or Integrated Gradients are rejected.

!!!warning "Interpretation"
    Localization maps measure sensitivity of a concept score to input perturbations. They are
    not object boundaries or segmentation labels. RISE and Sobol can also produce different
    maps because they use different perturbation and aggregation strategies.

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

{{xplique.concepts.holistic_craft.ConceptLocalizer}}

## References

[^1]: [CRAFT: Concept Recursive Activation FacTorization for Explainability (2023).](https://arxiv.org/pdf/2211.10154.pdf)

[^2]: [A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation (2023).](https://arxiv.org/pdf/2306.07304.pdf)
