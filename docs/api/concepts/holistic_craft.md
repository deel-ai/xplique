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

Holistic CRAFT works with various object detection architectures through specialized latent extractors provided by the companion `xplique-adapters` package. This package is under construction in the DEEL AI organization and will be pip-installable soon; the detector examples below use its planned API:

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

3. **Estimate Importance**: Use any attribution methods available in Xplique (gradient-based, perturbation-based) to rank concept importance
4. **Visualize**: Generate concept heatmaps overlaid on images to show "what" and "where"

Like regular CRAFT, Holistic CRAFT requires splitting the model into two parts: $(g, h)$ such that $f(x) = (g \cdot h)(x)$. The model $g$ maps input to latent space (activation maps), and $h$ maps latent space to predictions. Concepts are extracted from these activation maps in latent space.

This split is implemented through three abstractions:

- **`LatentData`**: A container that holds the intermediate activations produced by $g$. It abstracts away framework-specific tensor formats, providing a unified interface for reading (`get_activations`) and writing (`set_activations`) activations, with the necessary shape conversions (e.g., channel-first to channel-last).

- **`LatentExtractor`**: Wraps both $g$ (`input_to_latent_model`) and $h$ (`latent_to_logit_model`). It orchestrates the full forward pass, batching, device management, and output formatting. The `TorchLatentExtractor` and `TfLatentExtractor` subclasses provide framework-specific implementations.

- **`LatentExtractorBuilder`**: A factory that constructs a `LatentExtractor` for a specific model architecture. It handles all the architecture-specific wiring (defining how to split the model, which layer to extract from, and how to format outputs) so that the rest of the CRAFT pipeline remains model-agnostic.


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

### Using a Different NMF Factorizer

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

`EncodedData` is the named tuple returned by `HolisticCraft.encode()`. It contains the
image-specific `latent_data` and its concept coefficients, `coeffs_u`. `LatentData` is
the framework-independent interface used to read and replace intermediate activations.

The public core API consists of `HolisticCraft`, `PartialExplainer`, and `EncodedData`
from the `xplique.concepts` package. The framework-independent `LatentData`,
`LatentExtractor`, and `LatentExtractorBuilder` classes are available from
`xplique.concepts.latent_extractor`. `PartialExplainer` defers attribution-explainer
construction until a model and batch size are available.

### TensorFlow

`HolisticCraftTf` is the TensorFlow implementation. `TfLatentExtractor` provides the
TensorFlow latent extraction and decoding interface.

For generic layered TensorFlow models, use `LayeredModelExtractorBuilder` from
`xplique.concepts.tf.layered_model_latent_extractor`.

### PyTorch

`HolisticCraftTorch` is the PyTorch implementation. `TorchLatentData` stores the
framework-specific activations, and `TorchLatentExtractor` handles PyTorch latent
extraction and decoding.

For generic layered PyTorch models, use `LayeredModelExtractorBuilder` from
`xplique.concepts.torch.layered_model_latent_extractor`. The PyTorch-specific
`TorchSklearnNMFFactorizer` and optional `OvercompleteFactorizer` are available from
`xplique.concepts.torch.factorizer`.

## References

[^1]: [CRAFT: Concept Recursive Activation FacTorization for Explainability (2023).](https://arxiv.org/pdf/2211.10154.pdf)

[^2]: [A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation (2023).](https://arxiv.org/pdf/2306.07304.pdf)
