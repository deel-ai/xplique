"""
Unified Attribution Script for Object Detection Models
Supports: DETR, Faster R-CNN, RetinaNet, FCOS, SSD, YOLO
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import tensorflow as tf

import xplique
from xplique.attributions import Occlusion, Saliency
from xplique.wrappers import TorchWrapper

from xplique.plots.display_image_with_boxes import display_image_with_boxes
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager
from xplique.utils_functions.common.torch.gradients_check import check_model_gradients

# ==============================================================================
# Model Configurations
# ==============================================================================

MODEL_CONFIGS = {
    "detr": {
        "wrapper_class": "DetrBoxesModelWrapper",
        "wrapper_args": {},
        "box_format": BoxFormat.XYXY,
        "normalized": True,
        "confidence": 0.5,
        "use_weights_transform": False,
        "custom_normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "image_size": (800, 800),
        "has_latent_extractor": True,
    },
    "fasterrcnn": {
        "wrapper_class": "TorchvisionBoxesModelWrapper",
        "wrapper_args": {"nb_classes": 91},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.5,
        "use_weights_transform": True,
        "image_size": None,  # Use original size
    },
    "retinanet": {
        "wrapper_class": "TorchvisionBoxesModelWrapper",
        "wrapper_args": {"nb_classes": 91},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.5,
        "use_weights_transform": True,
        "image_size": None,  # Use original size
    },
    "fcos": {
        "wrapper_class": "TorchvisionBoxesModelWrapper",
        "wrapper_args": {"nb_classes": 91},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.5,
        "use_weights_transform": True,
        "image_size": None,  # Use original size
    },
    "ssd": {
        "wrapper_class": "TorchvisionBoxesModelWrapper",
        "wrapper_args": {"nb_classes": 91},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.15,  # Lower threshold for SSD
        "use_weights_transform": True,
        "image_size": None,  # Use original size
    },
    "yolo": {
        "wrapper_class": "YoloResultBoxesModelWrapper",
        "wrapper_args": {},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.35,
        "use_weights_transform": False,
        "image_size": (640, 640),
        "has_raw_wrapper": True,
        "raw_wrapper_class": "YoloRawBoxesModelWrapper",
        "model_path": "yolo11n.pt",
    },
    "yolo_relu": {
        "wrapper_class": "YoloResultBoxesModelWrapper",
        "wrapper_args": {},
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "confidence": 0.35,
        "use_weights_transform": False,
        "image_size": (640, 640),
        "has_raw_wrapper": True,
        "raw_wrapper_class": "YoloRawBoxesModelWrapper",
        "model_path": "best_yolo.pt",
    },
}

# COCO classes with N/A (91 classes - for DETR, RetinaNet, FCOS, SSD)
COCO_CLASSES_91 = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# COCO classes without N/A (80 classes - for YOLO)
COCO_CLASSES_80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attribution explanations for object detection models"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["detr", "fasterrcnn", "retinanet", "fcos", "ssd", "yolo", "yolo_relu"],
        help="Model to use"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Classes to generate attributions for (default: person car for most models)"
    )
    parser.add_argument(
        "--nb-boxes-to-explain", type=int, default=1,
        help=("Enable individual box explanations (default: 1 to enable, 0 to disable). "
              "When enabled, explains box 1 and box 2 per class")
    )
    parser.add_argument("--methods", nargs="+", default=["occlusion", "saliency"],
                        choices=["occlusion", "saliency"],
                        help="Attribution methods to use (default: occlusion saliency)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Confidence threshold for filtering boxes (overrides model defaults)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: torch_{model}_attributions)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: img.jpg for most models)")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only generate detection visualizations, skip attribution generation")
    return parser.parse_args()

# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(model_name, config, device):
    """Load object detection model."""
    print(f"Loading {model_name.upper()} model...")

    if model_name == "detr":
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        model.eval()
        model = model.to(device)
        return model, None

    elif model_name == "fasterrcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights
        )
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        model.eval()
        model = model.to(device)
        return model, weights

    elif model_name == "retinanet":
        from torchvision.models.detection import (
            retinanet_resnet50_fpn_v2,
            RetinaNet_ResNet50_FPN_V2_Weights
        )
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
        model.eval()
        model = model.to(device)
        return model, weights

    elif model_name == "fcos":
        from torchvision.models.detection import (
            fcos_resnet50_fpn,
            FCOS_ResNet50_FPN_Weights
        )
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(weights=weights, box_score_thresh=0.9)
        model.eval()
        model = model.to(device)
        return model, weights

    elif model_name == "ssd":
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights
        )
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights)
        model.eval()
        model = model.to(device)
        return model, weights

    elif model_name in ["yolo", "yolo_relu"]:
        from ultralytics import YOLO
        import warnings
        # Suppress YOLO verbose output
        warnings.filterwarnings('ignore')
        model_path = config.get("model_path", "yolo11n.pt")
        model = YOLO(model_path, verbose=False)
        model.eval()
        return model, None

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ==============================================================================
# Model Wrapper Creation
# ==============================================================================

def create_wrapper(model_name, model, config, device, use_raw_wrapper=False):
    """Create appropriate wrapper for the model."""

    if use_raw_wrapper:
        # Special handling for models with alternative extractors
        if model_name == "detr":
            from xplique_adapters.concepts.torch.latent_data_detr import DetrExtractorBuilder
            return DetrExtractorBuilder.build(model, device=str(device))

        elif model_name in ["yolo", "yolo_relu"]:
            from xplique_adapters.concepts.torch.latent_data_yolo import YoloExtractorBuilder
            detection_model = model.model
            batch_size = config.get("batch_size", 1)
            extraction_layer = config.get("extraction_layer", 10)
            return YoloExtractorBuilder.build(
                detection_model,
                extraction_layer=extraction_layer,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Raw wrapper not supported for model: {model_name}")

    else:
        # Standard wrappers (Phase 1)
        from xplique.utils_functions.object_detection.torch import box_model_wrapper
        wrapper_class_name = config["wrapper_class"]
        wrapper_class = getattr(box_model_wrapper, wrapper_class_name)
        wrapper_args = config.get("wrapper_args", {})
        return wrapper_class(model, **wrapper_args)

# ==============================================================================
# Image Processing
# ==============================================================================

def load_and_preprocess_image(image_path, model_name, config, weights, device):
    """Load and preprocess image according to model requirements."""
    print(f"Loading image from {image_path}...")

    # Load image
    if not os.path.exists(image_path):
        print("Image not found. Downloading default image...")
        os.system(
            'wget -O img.jpg "https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640"'
        )
        image_path = "img.jpg"

    image = Image.open(image_path).convert('RGB')

    # Resize if needed
    if config["image_size"] is not None:
        image = image.resize(config["image_size"])

    # Convert to tensor
    from torchvision.transforms.functional import to_tensor
    torch_image = to_tensor(image)
    visualizable_input = torch_image.unsqueeze(0)

    # Apply preprocessing
    if config["use_weights_transform"]:
        preprocess = weights.transforms()
        processed_input = preprocess(visualizable_input).to(device)
    elif "custom_normalize" in config:
        transform = T.Compose([
            T.Normalize(config["custom_normalize"][0], config["custom_normalize"][1])
        ])
        processed_input = transform(visualizable_input).to(device)
    else:
        # YOLO - use raw tensors
        processed_input = visualizable_input.to(device)

    return image, visualizable_input, processed_input

# ==============================================================================
# Attribution Generation
# ==============================================================================

def explain_and_visualize_with_boxes(
    explainer,
    inputs,
    boxes_cpu,
    box_manager,
    classes_list,
    label_to_color,
    display_image,
    output_filename,
    description
):
    """Generate explanations and overlay with bounding boxes."""
    print(f"  Generating {description}...")
    boxes_ext = tf.expand_dims(boxes_cpu, axis=0)
    explanations = explainer.explain(inputs, boxes_ext)

    # Generate heatmap
    from xplique.plots.image import generate_heatmap
    heatmap = generate_heatmap(
        explanations[0],
        size=inputs[0].shape[:2],
        clip_percentile=0.5
    )

    # Use display_image (PIL Image) for visualization with boxes
    # The heatmap is generated from inputs but displayed on the original image
    display_image_with_boxes(
        display_image,
        boxes_cpu,
        box_manager,
        classes_list,
        label_to_color,
        heatmap=heatmap
    )
    plt.savefig(output_filename)
    print(f"  Saved: {output_filename}")
    plt.close()

def generate_attributions_for_class(
    explainer,
    inputs,
    boxes_cpu,
    class_name,
    method_name,
    nb_boxes_to_explain,
    output_dir,
    file_offset,
    box_manager,
    classes_list,
    label_to_color,
    display_image
):
    """Generate attributions for all boxes and individual boxes of a class."""
    print(f"\n  Processing {class_name}...")

    # Skip if no boxes detected
    if boxes_cpu.shape[0] == 0:
        print(f"  No {class_name} boxes detected. Skipping...")
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"No {class_name} boxes detected",
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        plt.savefig(f'{output_dir}/{file_offset:02d}_{method_name}_{class_name}_all.png')
        plt.close()
        return file_offset + 1

    current_file = file_offset

    # Explanation for all boxes
    output_file = f'{output_dir}/{current_file:02d}_{method_name}_{class_name}_all.png'
    explain_and_visualize_with_boxes(
        explainer, inputs, boxes_cpu, box_manager, classes_list, label_to_color,
        display_image, output_file, f"all {class_name} boxes"
    )
    current_file += 1

    # Explanations for individual boxes
    total_boxes = boxes_cpu.shape[0]
    boxes_to_explain = []  # List of tuples: (idx, label, description)

    if nb_boxes_to_explain > 0 and total_boxes >= 1:
        # Extract scores (confidence values) from boxes
        # Box format: [x1, y1, x2, y2, confidence, class_id]
        scores = boxes_cpu[:, 4].numpy()

        # Add first box (person1)
        boxes_to_explain.append((0, "1", f"{class_name} box 1"))

        # Add second box (person2) if available
        if total_boxes >= 2:
            boxes_to_explain.append((1, "2", f"{class_name} box 2"))

        # Find box with maximum score (person_max)
        max_idx = int(np.argmax(scores))
        boxes_to_explain.append((max_idx, "max", f"{class_name} box with maximum score"))

        # Find box with minimum score (person_min)
        min_idx = int(np.argmin(scores))
        boxes_to_explain.append((min_idx, "min", f"{class_name} box with minimum score"))

    for idx, box_label, description in boxes_to_explain:
        box_single = boxes_cpu[idx:idx+1]
        output_file = f'{output_dir}/{current_file:02d}_{method_name}_{class_name}_{box_label}.png'
        explain_and_visualize_with_boxes(
            explainer, inputs, box_single, box_manager, classes_list, label_to_color,
            display_image, output_file, description
        )
        current_file += 1

    return current_file

# ==============================================================================
# Latent Extractor for DETR
# ==============================================================================



# ==============================================================================
# Main Function
# ==============================================================================

def main():
    args = parse_args()

    # Setup defaults based on model
    if args.classes is None:
        args.classes = ["person", "car"]

    if args.image is None:
        args.image = "img.jpg"

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = MODEL_CONFIGS[args.model]

    # Override confidence if provided
    if args.confidence is not None:
        config["confidence"] = args.confidence

    # Get class names (needed for output directory naming)
    CLASSES = COCO_CLASSES_80 if args.model in ["yolo", "yolo_relu"] else COCO_CLASSES_91

    # Validate requested classes
    for class_name in args.classes:
        if class_name not in CLASSES:
            print(f"WARNING: Class '{class_name}' not in model's class list. Skipping.")
            args.classes.remove(class_name)

    if not args.classes:
        print("ERROR: No valid classes specified.")
        return

    print(f"Classes to explain: {args.classes}")

    # Setup output directory (include class names)
    if args.output_dir is None:
        # Create a suffix with all class names
        classes_suffix = "_".join(args.classes)
        confidence_suffix = "" if args.confidence is None else f"_conf{args.confidence}"
        output_dir = f"torch_{args.model}_attributions_{classes_suffix}{confidence_suffix}"
    else:
        output_dir = args.output_dir

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load model
    model, weights = load_model(args.model, config, device)
    print("Model loaded successfully")

    # Load and preprocess image
    image, _, processed_input = load_and_preprocess_image(
        args.image, args.model, config, weights, device
    )
    print(f"Image shape: {processed_input.shape}")

    # Create model wrapper (standard wrapper for Phase 1)
    od_model = create_wrapper(args.model, model, config, use_raw_wrapper=False)

    # Get predictions
    results = od_model(processed_input)

    # Setup box manager
    box_manager = TorchBoxManager(config["box_format"], normalized=config["normalized"])

    # Visualize all detections
    print("\n" + "="*80)
    print("Visualizing Detections")
    print("="*80)

    filtered_results = results[0].filter(confidence=config["confidence"])
    print(f"Number of detections: {filtered_results.shape[0]}")

    # Color mapping for bounding boxes - dynamically set based on classes being studied
    # Default colors for common COCO classes
    all_colors = {
        'person': 'r',
        'car': "#03FFF2D1",
        'motorcycle': 'y',
        'bicycle': 'b'
    }

    # Build label_to_color with only the classes being studied
    label_to_color = {}
    default_colors = ['r', 'orange', 'b', 'g', 'y', 'purple', 'cyan', 'magenta']
    for idx, class_name in enumerate(args.classes):
        if class_name in all_colors:
            label_to_color[class_name] = all_colors[class_name]
        else:
            # Assign a default color if class not in predefined mapping
            label_to_color[class_name] = default_colors[idx % len(default_colors)]

    display_image = image.resize(config["image_size"]) if config["image_size"] else image
    if config["normalized"]:
        display_image = image
    # display_image = image.resize(config["image_size"]) if (config["image_size"] and not config["normalized"]) else image

    fig = display_image_with_boxes(
        display_image,
        filtered_results.cpu().detach(),
        box_manager,
        CLASSES,
        label_to_color
    )
    fig.savefig(f'{OUTPUT_DIR}/01_{args.model}_detections.png')
    print(f"Saved: 01_{args.model}_detections.png")
    plt.close(fig)

    # Filter boxes by class and visualize
    print("\n" + "="*80)
    print("Filtering Boxes by Class")
    print("="*80)

    class_boxes = {}
    file_num = 2

    for class_name in args.classes:
        class_id = CLASSES.index(class_name)
        boxes = results[0].filter(class_id=class_id, confidence=config["confidence"])
        # Store GPU tensors directly - conversion to CPU happens only when needed
        # display_image_with_boxes() handles conversion internally via box_manager.to_numpy_tuple()
        class_boxes[class_name] = boxes

        print(f"\n{class_name}: {boxes.shape[0]} boxes")

        # Visualize boxes for this class
        # Note: display_image_with_boxes() accepts GPU tensors and converts internally
        if boxes.shape[0] > 0:
            fig = display_image_with_boxes(
                display_image,
                boxes,
                box_manager,
                CLASSES,
                label_to_color
            )
        else:
            # Create placeholder
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(display_image)
            ax.text(0.5, 0.5, f"No {class_name} boxes detected",
                    ha='center', va='center', fontsize=16, color='white',
                    bbox={'boxstyle': 'round', 'facecolor': 'red', 'alpha': 0.7})
            ax.axis('off')

        fig.savefig(f'{OUTPUT_DIR}/{file_num:02d}_{class_name}_boxes.png')
        print(f"Saved: {file_num:02d}_{class_name}_boxes.png")
        plt.close(fig)
        file_num += 1

    # Early exit if only detection visualization is requested
    if args.detect_only:
        print("\n" + "="*80)
        print("DETECT-ONLY MODE: Skipping attribution generation")
        print("="*80)
        print(f"Detection outputs saved to: {OUTPUT_DIR}")
        return

    # ==========================================================================
    # PHASE 1: Standard Wrapper (all models)
    # ==========================================================================
    # Always run Phase 1 first to attempt standard attributions

    current_file_num = file_num
    phase1_successful = {"occlusion": False, "saliency": False}

    print("\n" + "="*80)
    print("PHASE 1: Standard Wrapper Attributions")
    print("="*80)

    # Check gradients
    print("\nChecking Model Gradients...")
    try:
        gradient_check = check_model_gradients(od_model, processed_input)
        print(f"Gradient check: {'PASSED' if gradient_check else 'FAILED'}")
    except Exception as e:
        print(f"Gradient check failed: {e}")
        gradient_check = False

    # Setup for attribution
    od_model.set_output_as_tensor()
    # Note: Don't call eval() on YOLO wrapper as it triggers training mode initialization
    if hasattr(od_model, 'eval') and args.model not in ["yolo", "yolo_relu"]:
        od_model.eval()

    # Convert to TensorFlow format
    tf_inputs = tf.convert_to_tensor(processed_input.cpu().numpy())
    tf_inputs = tf.transpose(tf_inputs, [0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
    print(f"TensorFlow inputs shape: {tf_inputs.shape}")

    # Generate attributions for each method
    for method in args.methods:
        print("\n" + "="*80)
        print(f"Generating {method.upper()} Attributions (Phase 1)")
        print("="*80)

        try:
            # Create wrapper and explainer based on method
            # Saliency needs requires_grad=True, Occlusion needs requires_grad=False
            if method == "occlusion":
                torch_wrapped_model = TorchWrapper(od_model, device=device, requires_grad=False)
                explainer = Occlusion(
                    torch_wrapped_model,
                    operator=xplique.Tasks.OBJECT_DETECTION,
                    batch_size=1,
                    patch_size=(100, 100),
                    patch_stride=(50, 50)
                )
            elif method == "saliency":
                if not gradient_check:
                    print("WARNING: Skipping Saliency due to gradient check failure")
                    if args.model == "detr":
                        print(
                            "NOTE: This is expected for DETR - "
                            "will use latent extractor in Phase 2"
                        )
                    elif args.model in ["yolo", "yolo_relu"]:
                        print(
                            "NOTE: This is expected for YOLO - "
                            "will use raw wrapper in Phase 2"
                        )
                    continue

                torch_wrapped_model = TorchWrapper(od_model, device=device, requires_grad=True)
                explainer = Saliency(
                    torch_wrapped_model,
                    operator=xplique.Tasks.OBJECT_DETECTION,
                    batch_size=1
                )
            else:
                print(f"Unknown method: {method}")
                continue

            # Generate attributions for each class
            method_succeeded = True
            for class_name in args.classes:
                boxes = class_boxes[class_name]
                boxes_cpu = boxes.cpu().detach()

                try:
                    current_file_num = generate_attributions_for_class(
                        explainer,
                        tf_inputs,
                        boxes_cpu,
                        class_name,
                        method,
                        args.nb_boxes_to_explain,
                        OUTPUT_DIR,
                        current_file_num,
                        box_manager,
                        CLASSES,
                        label_to_color,
                        display_image
                    )
                except Exception as e:
                    print(f"ERROR: {method} failed for {class_name}: {e}")
                    if args.model in ["detr", "yolo"]:
                        print(
                            f"NOTE: This is expected for {args.model.upper()} - "
                            "will retry in Phase 2"
                        )
                    method_succeeded = False
                    break

            if method_succeeded:
                phase1_successful[method] = True

        except Exception as e:
            print(f"ERROR: {method} explainer creation failed: {e}")
            if args.model in ["detr", "yolo"]:
                print(
                    f"NOTE: This is expected for {args.model.upper()} - "
                    "will retry in Phase 2"
                )

    # ==========================================================================
    # PHASE 2: Alternative Wrapper (DETR and YOLO only)
    # ==========================================================================
    # Always run Phase 2 for DETR and YOLO when any Phase 1 method failed
    needs_phase2 = (args.model == "detr" or args.model in ["yolo", "yolo_relu"])
    run_phase2 = needs_phase2 and not all(phase1_successful.values())

    if run_phase2:
        print("\n" + "="*80)
        print("PHASE 2: Alternative Wrapper Attributions")
        print("="*80)

        # Create alternative wrapper using create_wrapper with use_raw_wrapper=True
        print(f"\nBuilding {args.model.upper()} alternative wrapper...")
        alt_model = create_wrapper(args.model, model, config, device, use_raw_wrapper=True)
        alt_model.set_output_as_tensor()

        # Set suffix based on model type
        suffix = "_latent" if args.model == "detr" else "_raw"
        print(f"{args.model.upper()} alternative wrapper built successfully!")

        # Prepare inputs and wrapped model based on model type
        alt_class_boxes = {}  # Initialize for YOLO, unused for DETR
        if args.model == "detr":
            # DETR: Wrapper requires specific parameters for latent extractor
            alt_wrapped_model = TorchWrapper(
                alt_model,
                device=device,
                requires_grad=True,
                is_channel_first=True
            )
            # Input format: channel-last numpy array
            alt_inputs = processed_input.cpu().detach().numpy().transpose(0, 2, 3, 1)

        elif args.model in ["yolo", "yolo_relu"]:
            # YOLO: Set training flag to False for TorchWrapper compatibility
            alt_model.training = False
            alt_wrapped_model = TorchWrapper(alt_model, device=device)

            # Input format: TF tensor channel-last
            alt_inputs = tf.convert_to_tensor(processed_input.cpu().numpy())
            alt_inputs = tf.transpose(alt_inputs, [0, 2, 3, 1])

            # Get predictions from alternative wrapper for YOLO
            alt_results = alt_model(processed_input)
            # Re-filter boxes for each class with the new wrapper
            alt_class_boxes = {}
            for class_name in args.classes:
                class_id = CLASSES.index(class_name)
                boxes = alt_results[0].filter(class_id=class_id, confidence=config["confidence"])
                alt_class_boxes[class_name] = boxes
                print(f"{class_name}: {boxes.shape[0]} boxes (raw wrapper)")
        else:
            # Should not reach here since needs_phase2 checks for detr/yolo
            raise ValueError(f"Phase 2 requested for unsupported model: {args.model}")

        print(f"\nInput shape for Phase 2: {alt_inputs.shape}")

        # Generate attributions with alternative wrapper
        for method in args.methods:
            print("\n" + "="*80)
            print(f"Generating {method.upper()} Attributions (Phase 2 - {suffix[1:]})")
            print("="*80)

            # Create explainer
            if method == "occlusion":
                explainer = Occlusion(
                    alt_wrapped_model,
                    operator=xplique.Tasks.OBJECT_DETECTION,
                    batch_size=1,
                    patch_size=(100, 100),
                    patch_stride=(50, 50)
                )
            elif method == "saliency":
                explainer = Saliency(
                    alt_wrapped_model,
                    operator=xplique.Tasks.OBJECT_DETECTION,
                    batch_size=1
                )
            else:
                print(f"Unknown method: {method}")
                continue

            # Generate attributions for each class
            for class_name in args.classes:
                # Use alternative boxes for YOLO, original for DETR
                if args.model in ["yolo", "yolo_relu"]:
                    boxes = alt_class_boxes[class_name]
                else:
                    boxes = class_boxes[class_name]

                boxes_cpu = boxes.cpu().detach()

                # Generate with suffix in method name for proper file naming
                current_file_num = generate_attributions_for_class(
                    explainer,
                    alt_inputs,
                    boxes_cpu,
                    class_name,
                    f"{method}{suffix}",
                    args.nb_boxes_to_explain,
                    OUTPUT_DIR,
                    current_file_num,
                    box_manager,
                    CLASSES,
                    label_to_color,
                    display_image
                )

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print(f"All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
