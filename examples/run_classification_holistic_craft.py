"""
Example script demonstrating Holistic CRAFT on image classification models.

This script shows how to use CRAFT to extract and analyze concepts from
image classification models (ResNet, VGG) and rank them by importance.
"""

import os
import argparse
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from horama import maco, plot_maco
from tensorflow.keras.utils import to_categorical

import xplique
from xplique.concepts import HolisticCraftTorch as CraftTorch
from xplique.concepts.torch.latent_data_layered_model import LayeredModelExtractorBuilder
from xplique.attributions import GradientInput, IntegratedGradients
from xplique.plots import plot_attributions
from xplique.wrappers import TorchWrapper


MODEL_CONFIGS = {
    "resnet50": {
        "model_class": models.resnet50,
        "image_size": (224, 224),
        "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "split_layer": -2,  # Split at avgpool (before fc layer)
        "num_classes": 1000,
    },
    "resnet18": {
        "model_class": models.resnet18,
        "image_size": (224, 224),
        "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "split_layer": -2,  # Split at avgpool (before fc layer)
        "num_classes": 1000,
    },
    "vgg16": {
        "model_class": models.vgg16,
        "image_size": (224, 224),
        "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "split_layer": -2,  # Split before final classifier
        "num_classes": 1000,
    },
}


def parse_args():
    """Parse command-line arguments for CRAFT on image classification models."""
    parser = argparse.ArgumentParser(description="Run CRAFT on image classification models")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "resnet18", "vgg16"],
                        help="Model to use (default: resnet50)")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "rabbit"],
                        help="Dataset to use: 'imagenet' for custom images, "
                             "'rabbit' for rabbit dataset (default: imagenet)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (imagenet mode) or index (rabbit mode). "
                             "Default: 'img.jpg' for imagenet, 0 for rabbit")
    parser.add_argument("--num_concepts", type=int, default=10,
                        help="Number of concepts for CRAFT (default: 10)")
    parser.add_argument("--split_layer", type=int, default=-2,
                        help="Layer index for feature extraction (default: model-specific)")
    parser.add_argument("--class_id", type=int, default=None,
                        help="ImageNet class ID to analyze (default: use top prediction)")
    parser.add_argument(
        "--num_images_sobol", type=int, default=1,
        help="Number of images for Sobol attribution (default: 1, 0 to skip)"
    )
    parser.add_argument(
        "--num_images_gradient_input", type=int, default=1,
        help="Number of images for Gradient Input attribution (default: 1)"
    )
    parser.add_argument("--skip-feature-viz", action="store_true",
                        help="Skip feature visualization step")
    parser.add_argument("--feature-viz-nb-top-concepts", type=int, default=4,
                        help="Number of top concepts to visualize (default: 4)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument(
        "--display-image", type=str, default=None,
        help="Path to image for displaying concepts (imagenet mode) or index "
             "(rabbit mode). If not specified, uses the same as --image. "
             "This controls which image(s) to display concepts on in section 7.")

    return parser.parse_args()


def load_imagenet_classes():
    """Load ImageNet class labels."""
    # Try to load from file if available
    if os.path.exists("imagenet_classes.txt"):
        with open("imagenet_classes.txt", "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    print("Warning: imagenet_classes.txt not found. Using class IDs instead.")
    return [f"Class {i}" for i in range(1000)]


def load_and_preprocess_image(image_path, config):
    """Load and preprocess a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_image = Image.open(image_path).convert('RGB')

    # Standard ImageNet preprocessing
    transform = T.Compose([
        T.Resize(config["image_size"]),
        T.ToTensor(),
        T.Normalize(config["normalize"][0], config["normalize"][1])
    ])

    input_tensor = transform(raw_image).unsqueeze(0)

    return raw_image, input_tensor


def load_model(model_name, config, device):
    """Load a pretrained classification model."""
    print(f"Loading {model_name} model...")
    model = config["model_class"](pretrained=True).to(device)
    model.eval()
    print(f"{model_name} model loaded successfully")
    return model


def build_latent_extractor(model, config, device):
    """Build latent extractor for the model."""
    split_layer = config["split_layer"]

    latent_extractor = LayeredModelExtractorBuilder.build(
        model=model,
        split_layer=split_layer,
        device=str(device),
        batch_size=1
    )

    print(f"Latent extractor built successfully (split_layer={split_layer})")
    return latent_extractor



def load_rabbit_images(config, device):
    """
    Load rabbit images from assets for CRAFT training.

    Returns:
        tuple: (preprocessed_tensors, raw_images_array)
            - preprocessed_tensors: torch.Tensor of shape (N, C, H, W), normalized
            - raw_images_array: numpy.ndarray of shape (N, H, W, 3), values in [0, 255]
    """
    rabbit_path = os.path.join(os.path.dirname(__file__), "assets", "rabbit.npz")

    if not os.path.exists(rabbit_path):
        print(f"Warning: {rabbit_path} not found.")
        return None, None

    # Load rabbit images
    data = np.load(rabbit_path)
    rabbit_images_raw = data['arr_0']  # Shape: (300, 224, 224, 3), values in [0, 255]

    print(f"Loaded {len(rabbit_images_raw)} rabbit images from {rabbit_path}")
    print(f"Rabbit images shape: {rabbit_images_raw.shape}")

    # Normalize using ImageNet stats (same as input image preprocessing)
    mean = np.array(config["normalize"][0]).reshape(1, 1, 1, 3)
    std = np.array(config["normalize"][1]).reshape(1, 1, 1, 3)

    # Convert to [0, 1] range first, then normalize
    rabbit_images_normalized = (rabbit_images_raw / 255.0 - mean) / std

    # Convert to PyTorch format (channels first) and create tensor
    rabbit_images_torch = torch.from_numpy(rabbit_images_normalized).float()
    rabbit_images_torch = rabbit_images_torch.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    rabbit_images_torch = rabbit_images_torch.to(device)

    print(f"Preprocessed rabbit images shape: {rabbit_images_torch.shape}")

    return rabbit_images_torch, rabbit_images_raw


def get_rabbit_image_by_index(rabbit_images_raw, rabbit_images_torch, index, config, device):
    """
    Get a single rabbit image by index.

    Args:
        rabbit_images_raw: numpy array of raw images (N, H, W, 3) in [0, 255]
        rabbit_images_torch: torch tensor of preprocessed images (N, C, H, W)
        index: index of the image to retrieve
        config: model config (unused, for API consistency)
        device: torch device (unused, for API consistency)

    Returns:
        tuple: (raw_image_pil, input_tensor)
            - raw_image_pil: PIL Image
            - input_tensor: torch.Tensor of shape (1, C, H, W), normalized
    """
    # pylint: disable=unused-argument
    if index < 0 or index >= len(rabbit_images_raw):
        raise ValueError(
            f"Image index {index} out of range [0, {len(rabbit_images_raw)-1}]")

    # Get raw image and convert to PIL
    raw_image_np = rabbit_images_raw[index].astype(np.uint8)
    raw_image_pil = Image.fromarray(raw_image_np)

    # Get preprocessed tensor (add batch dimension)
    input_tensor = rabbit_images_torch[index:index+1]

    return raw_image_pil, input_tensor


def plot_concepts_feature_viz(
        craft, input_tensor, indices_to_plot, device, model_input_size,
        output_dir, suffix="", file_number=10):
    """Generate feature visualizations for top concepts."""
    def objective(images, concept_id=0):
        latent_data_and_coeffs_u = craft.encode(images, differentiable=True)
        coeffs_u_list = [pair.coeffs_u for pair in latent_data_and_coeffs_u]
        coeffs_u = torch.cat(coeffs_u_list)
        # For classification, coeffs_u might be 2D (batch, concepts) or 4D (batch, h, w, concepts)
        if len(coeffs_u.shape) == 4:
            return torch.mean(coeffs_u[:, :, :, concept_id])
        return torch.mean(coeffs_u[:, concept_id])

    def objective_for_concept(concept_id):
        return partial(objective, concept_id=concept_id)

    n = len(indices_to_plot)
    _, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for i, idx in enumerate(indices_to_plot):
        image1, alpha1 = maco(objective_for_concept(idx), model_input_size=model_input_size,
                             crops_per_iteration=2, device=device, total_steps=100,
                             noise=0.5, box_size=(0.30, 0.35))
        ax = axes[i] if n > 1 else axes
        plt.sca(ax)
        plot_maco(image1, alpha1)
        ax.set_title(f"Concept {idx}")

    plt.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    output_path = os.path.join(output_dir, f"{file_number:02d}_feature_viz{suffix_str}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Run CRAFT concept extraction and analysis on image classification models."""
    args = parse_args()

    config = MODEL_CONFIGS[args.model]

    # Override split_layer if provided via CLI
    if args.split_layer is not None:
        config = config.copy()
        config["split_layer"] = args.split_layer

    # Set default image based on dataset mode
    if args.image is None:
        if args.dataset == "rabbit":
            image_arg = "0"  # Default to first rabbit image
        else:
            image_arg = "img.jpg"  # Default to img.jpg for imagenet
    else:
        image_arg = args.image

    # Set display_image (for concept visualization)
    if args.display_image is None:
        display_image_arg = image_arg  # Use same as --image by default
    else:
        display_image_arg = args.display_image

    output_dir_name = f"torch_{args.model}_craft_classification_{args.dataset}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset mode: {args.dataset}")

    print("\n" + "="*80)
    print("1. Loading and Preprocessing Image")
    print("="*80)

    # Load images based on dataset mode
    rabbit_images_raw = None  # Initialize for later use
    if args.dataset == "rabbit":
        # Load all rabbit images first
        rabbit_images_torch, rabbit_images_raw = load_rabbit_images(config, device)

        if rabbit_images_torch is None:
            raise FileNotFoundError(
                "Rabbit dataset not found. Please ensure assets/rabbit.npz exists.")

        # Parse image_arg as index
        try:
            image_index = int(image_arg)
        except ValueError as exc:
            raise ValueError(
                f"In rabbit mode, --image must be an integer index, got: {image_arg}") from exc

        # Get the selected rabbit image
        raw_image, input_tensor = get_rabbit_image_by_index(
            rabbit_images_raw, rabbit_images_torch, image_index, config, device)

        print(f"Selected rabbit image index: {image_index} out of {len(rabbit_images_raw)}")
        craft_training_data = rabbit_images_torch  # Use all rabbit images for CRAFT

    else:  # imagenet mode
        # Load single image from path
        raw_image, input_tensor = load_and_preprocess_image(image_arg, config)
        input_tensor = input_tensor.to(device)
        print(f"Image loaded: {image_arg}")
        craft_training_data = input_tensor  # Use single image for CRAFT

    print(f"Image size: {raw_image.size}")
    print(f"Input tensor shape: {input_tensor.shape}")

    # Display the input image
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_image)
    plt.title("Input Image")
    plt.axis('off')
    output_path = os.path.join(output_dir, "01_input_image.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print(f"2. Loading {args.model.upper()} Model")
    print("="*80)

    model = load_model(args.model, config, device)

    # Load ImageNet classes
    imagenet_classes = load_imagenet_classes()

    print("\n" + "="*80)
    print("3. Getting Model Predictions")
    print("="*80)

    with torch.no_grad():
        predictions = model(input_tensor)

    # Get top-5 predictions
    probs = torch.nn.functional.softmax(predictions, dim=1)
    top5_prob, top5_catid = torch.topk(probs, 5)

    print("Top-5 predictions:")
    for i in range(5):
        class_id = int(top5_catid[0][i].item())
        prob = float(top5_prob[0][i].item())
        class_name = imagenet_classes[class_id]
        print(f"  {i+1}. {class_name} (ID: {class_id}): {prob:.4f}")

    # Determine which class to analyze
    if args.class_id is not None:
        target_class_id = int(args.class_id)
    else:
        target_class_id = int(top5_catid[0][0].item())

    target_class_name = imagenet_classes[target_class_id]
    print(f"\nAnalyzing class: {target_class_name} (ID: {target_class_id})")

    print("\n" + "="*80)
    print("4. Building Latent Extractor")
    print("="*80)

    latent_extractor = build_latent_extractor(model, config, device)

    # Test latent extractor
    with torch.no_grad():
        results = latent_extractor(input_tensor)
    print(f"Latent extractor output shape: {results.shape}")

    print("\n" + "="*80)
    print("5. Computing Attribution Maps for Target Class")
    print("="*80)

    # Convert input for Xplique (channels last)
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # Wrap model for Xplique
    torch_wrapped_model = TorchWrapper(latent_extractor, device=device, is_channel_first=True)

    # Create explainer
    from xplique.attributions import Saliency
    explainer = Saliency(torch_wrapped_model, batch_size=1)
    explanation = explainer.explain(input_tensor_tf_dim, targets=np.array([int(target_class_id)]))

    print(f"Explanation shape: {explanation.shape}")

    # Visualize saliency
    plot_attributions(explanation, [np.array(raw_image.resize(config["image_size"]))],
                     img_size=6., cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.75)

    output_path = os.path.join(output_dir, "02_saliency_map.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create Occlusion explainer
    from xplique.attributions import Occlusion
    explainer_occ = Occlusion(torch_wrapped_model,
                        operator=xplique.Tasks.CLASSIFICATION,
                        batch_size=1,
                        patch_size=(30, 30),
                        patch_stride=(15, 15))
    # Convert target to one-hot encoding for Occlusion
    targets_one_hot = to_categorical([int(target_class_id)], config["num_classes"])
    explanation_occ = explainer_occ.explain(
        input_tensor_tf_dim, targets=targets_one_hot)
    print(f"Occlusion explanation shape: {explanation_occ.shape}")

    # Visualize occlusion
    plot_attributions(explanation_occ, [np.array(raw_image.resize(config["image_size"]))],
                     img_size=6., cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.75)
    output_path = os.path.join(output_dir, "02b_occlusion_map.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create Gradient Input explainer
    explainer_gi = GradientInput(torch_wrapped_model, batch_size=1)
    explanation_gi = explainer_gi.explain(
        input_tensor_tf_dim, targets=np.array([int(target_class_id)]))
    print(f"Gradient Input explanation shape: {explanation_gi.shape}")

    # Visualize gradient input
    plot_attributions(explanation_gi, [np.array(raw_image.resize(config["image_size"]))],
                     img_size=6., cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.75)
    output_path = os.path.join(output_dir, "02c_gradient_input_map.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create Integrated Gradients explainer
    explainer_ig = IntegratedGradients(torch_wrapped_model, batch_size=1, steps=50)
    explanation_ig = explainer_ig.explain(
        input_tensor_tf_dim, targets=np.array([int(target_class_id)]))
    print(f"Integrated Gradients explanation shape: {explanation_ig.shape}")

    # Visualize integrated gradients
    plot_attributions(explanation_ig, [np.array(raw_image.resize(config["image_size"]))],
                     img_size=6., cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.75)
    output_path = os.path.join(output_dir, "02d_integrated_gradients_map.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print("6. Preparing CRAFT Training Data")
    print("="*80)

    if args.dataset == "rabbit" and rabbit_images_raw is not None:
        print(f"Using all {len(craft_training_data)} rabbit images for CRAFT training")

        # Display sample rabbit images
        num_samples = min(8*4, len(rabbit_images_raw))
        _, axes = plt.subplots(2*4, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(num_samples):
            # Use raw images directly (already in [0, 255] uint8 format)
            img_display = rabbit_images_raw[i] / 255.0  # Convert to [0, 1] for display

            axes[i].imshow(img_display)
            axes[i].axis('off')
            axes[i].set_title(f"Rabbit {i+1}")

        plt.tight_layout()
        output_path = os.path.join(output_dir, "03_rabbit_samples.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        print("Using single input image for CRAFT training (imagenet mode)")

    print("\n" + "="*80)
    print("7. Initializing and Running CRAFT")
    print("="*80)

    craft = CraftTorch(
        latent_extractor=latent_extractor,
        number_of_concepts=args.num_concepts,
        device=str(device)
    )

    # Fit CRAFT on the training data
    print(f"Fitting CRAFT on {len(craft_training_data)} images...")
    craft.fit(craft_training_data)
    print(f"CRAFT fitting completed with {args.num_concepts} concepts")

    print("\n" + "="*80)
    print("8. Loading Display Image for Concept Visualization")
    print("="*80)

    # Load the image to display concepts on (can be different from training image)
    if args.dataset == "rabbit":
        # Parse display_image_arg as index
        try:
            display_image_index = int(display_image_arg)
        except ValueError as exc:
            raise ValueError(
                f"In rabbit mode, --display-image must be an integer index, "
                f"got: {display_image_arg}") from exc

        # Get the selected rabbit image for display
        display_raw_image, display_input_tensor = get_rabbit_image_by_index(
            rabbit_images_raw, rabbit_images_torch, display_image_index, config, device)

        print(
            f"Display image index: {display_image_index} out of {len(rabbit_images_raw)}")
    else:  # imagenet mode
        # Load display image from path
        display_raw_image, display_input_tensor = load_and_preprocess_image(
            display_image_arg, config)
        display_input_tensor = display_input_tensor.to(device)
        print(f"Display image loaded: {display_image_arg}")

    print(f"Display image size: {display_raw_image.size}")
    print(f"Display input tensor shape: {display_input_tensor.shape}")

    print("\n" + "="*80)
    print("9. Displaying Concepts on Selected Image")
    print("="*80)

    craft.display_images_per_concept(
        display_input_tensor, order=None, filter_percentile=80, clip_percentile=5)

    output_path = os.path.join(output_dir, "04_concepts_unordered.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print(f"10. Ranking Concepts by Importance for '{target_class_name}' class")
    print("="*80)

    operator = xplique.Tasks.CLASSIFICATION

    # Evaluate with Gradient Input on display image
    print(f"\nRunning Gradient Input attribution with {args.num_images_gradient_input} image(s)...")
    importances_gi = craft.estimate_importance(
        display_input_tensor, operator, int(target_class_id), method='gradient_input')
    order_gi = importances_gi.argsort()[::-1]

    print(f"Concept importances (Gradient Input) for '{target_class_name}': {importances_gi}")
    print(f"Concept order: {order_gi}")

    # Conditionally evaluate with Sobol
    run_sobol = args.num_images_sobol > 0
    order_sobol = None
    importances_sobol = None
    if run_sobol:
        print(f"\nRunning Sobol attribution with {args.num_images_sobol} image(s)...")
        importances_sobol = craft.estimate_importance(
            display_input_tensor,
            operator,
            int(target_class_id),
            method="sobol",
            nb_design=32  # Number of Sobol samples
        )
        order_sobol = importances_sobol.argsort()[::-1]

        print(f"Concept importances (Sobol) for '{target_class_name}': {importances_sobol}")
        print(f"Concept order: {order_sobol}")
    else:
        print("\nSkipping Sobol attribution (num_images_sobol=0)")

    # Display concepts ordered by Gradient Input on display image
    craft.display_images_per_concept(
        display_input_tensor, order=order_gi.tolist(), filter_percentile=80, clip_percentile=5)

    output_path = os.path.join(output_dir, "05_concepts_ordered_gi.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Display concepts ordered by Sobol (if run) on display image
    if run_sobol and order_sobol is not None and importances_sobol is not None:
        craft.display_images_per_concept(
            display_input_tensor, order=order_sobol.tolist(), filter_percentile=80,
            clip_percentile=5)

        output_path = os.path.join(output_dir, "06_concepts_ordered_sobol.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    # Plot importance bars for Gradient Input
    plt.figure(figsize=(10, 6))
    plt.bar(list(range(len(importances_gi))), importances_gi)
    plt.xlabel('Concept ID')
    plt.ylabel('Importance')
    plt.title(f'Concept Importances (Gradient Input) for "{target_class_name}" class')

    output_path = os.path.join(output_dir, "07_importance_bars_gi.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Plot importance bars for Sobol (if run)
    if run_sobol and importances_sobol is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(list(range(len(importances_sobol))), importances_sobol)
        plt.xlabel('Concept ID')
        plt.ylabel('Importance')
        plt.title(f'Concept Importances (Sobol) for "{target_class_name}" class')

        output_path = os.path.join(output_dir, "08_importance_bars_sobol.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    print("\n" + "="*80)
    print("11. Computing Concept-wise Explanations")
    print("="*80)

    # Test compute_explanation_per_concept on display image
    explainer_partial = partial(
        GradientInput,
        harmonize=False,
    )
    explanation = craft.compute_explanation_per_concept(
        display_input_tensor, class_id=int(target_class_id),
        explainer_partial=explainer_partial)

    print(f"Explanation per concept shape: {explanation.shape}")
    print("  Expected: (batch, height, width, num_concepts)")

    # Visualize top concepts' spatial explanations
    nb_top = min(4, args.num_concepts)
    _, axes = plt.subplots(1, nb_top, figsize=(5 * nb_top, 5))
    if nb_top == 1:
        axes = [axes]

    display_raw_image_resized = np.array(display_raw_image.resize(config["image_size"]))

    for i in range(nb_top):
        concept_idx = order_gi[i]
        ax = axes[i]

        # Get explanation for this concept
        concept_explanation = explanation[0, :, :, concept_idx]

        # Display image with overlay
        ax.imshow(display_raw_image_resized)
        im = ax.imshow(concept_explanation, cmap='jet', alpha=0.5)
        ax.set_title(f"Concept {concept_idx}\n(Importance: {importances_gi[concept_idx]:.3f})")
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "09_concept_explanations.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print("12. Feature Visualization")
    print("="*80)

    if args.skip_feature_viz:
        print("Skipping feature visualization (--skip-feature-viz)")
    else:
        model_input_size = config["image_size"][0]
        nb_top_concepts = args.feature_viz_nb_top_concepts

        # Feature viz for Gradient Input
        plot_concepts_feature_viz(
            craft, input_tensor, order_gi[:nb_top_concepts], device,
            model_input_size, output_dir, suffix="gi", file_number=10)

        # Feature viz for Sobol (if run)
        if run_sobol and order_sobol is not None:
            plot_concepts_feature_viz(
                craft, input_tensor, order_sobol[:nb_top_concepts], device,
                model_input_size, output_dir, suffix="sobol", file_number=11)

    print("\n" + "="*80)
    print("All outputs saved to:", output_dir)
    print("="*80)
    print(f"Note: CRAFT was fitted on {len(craft_training_data)} images, "
          f"but concepts were displayed/ranked on the selected display image.")


if __name__ == "__main__":
    main()
