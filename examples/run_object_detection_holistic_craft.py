import _setup_path
import os
import argparse
from functools import partial
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
import xplique
from xplique.concepts import HolisticCraftTorch as CraftTorch
from xplique.concepts import HolisticCraftTf as CraftTf
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager
from xplique.utils_functions.object_detection.tf.box_manager import TfBoxManager
from xplique.plots.display_image_with_boxes import display_image_with_boxes
from horama import maco, plot_maco

MODEL_CONFIGS = {
    "detr": {
        "image_size": (800, 800),
        "use_weights_transform": False,
        "custom_normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "box_format": BoxFormat.XYXY,
        "normalized": True,
        "accuracy": 0.85,
        "default_num_images": 500,
        "default_start_offset": 0,
        "batch_size": 4,
    },
    "retinanet": {
        "image_size": (800, 800),
        "use_weights_transform": True,
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.3,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": -1,
    },
    "fasterrcnn": {
        "image_size": (800, 800),
        "use_weights_transform": True,
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.5,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": -1,
    },
    "fcos": {
        "image_size": (800, 800),
        "use_weights_transform": True,
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.5,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": -1,
    },
    "ssd": {
        "image_size": (320, 320),
        "use_weights_transform": True,
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.3,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": 2,
    },
    "yolo": {
        "model_path": "yolo11n.pt",
        "image_size": (640, 640),
        "use_weights_transform": False,
        "custom_transform": "yolo",
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.5,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": 10,
        "batch_size": 1,
    },
    "yolo_relu": {
        "model_path": "best_yolo.pt",
        "image_size": (640, 640),
        "use_weights_transform": False,
        "custom_transform": "yolo",
        "box_format": BoxFormat.XYXY,
        "normalized": False,
        "accuracy": 0.5,
        "default_num_images": 100,
        "default_start_offset": 0,
        "extraction_layer": 10,
        "batch_size": 1,
    },
}

COCO_CLASSES = [
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

PASCAL_VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CRAFT on object detection models")
    parser.add_argument("--framework", type=str, default="torch",
                        choices=["torch", "tf"],
                        help="Framework to use: 'torch' or 'tf' (default: torch)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["detr", "retinanet", "fasterrcnn", "fcos", "ssd", "yolo", "yolo_relu"],
                        help="Model to use")
    parser.add_argument("--class_name", type=str, default="giraffe",
                        help="COCO class name to analyze")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to load (default: model-specific)")
    parser.add_argument("--start_offset", type=int, default=None,
                        help="Starting index in COCO dataset (default: model-specific)")
    parser.add_argument("--num_concepts", type=int, default=10,
                        help="Number of concepts for CRAFT")
    parser.add_argument("--nmf_type", type=str, default="sklearn",
                        choices=["sklearn", "oc_semi_nmf"],
                        help="NMF factorizer type")
    parser.add_argument("--extraction_layer", type=int, default=None,
                        help="Layer index for feature extraction (default: model-specific)")
    parser.add_argument("--extract_location", type=str, default="resnet",
                        choices=["resnet", "fpn"],
                        help="Extraction location for FPN-based models (fasterrcnn/retinanet/fcos): 'resnet' or 'fpn' (default: resnet)")
    parser.add_argument("--num_images_sobol", type=int, default=5,
                        help="Number of images to use for Sobol attribution (default: 5, 0 to skip)")
    parser.add_argument("--skip-feature-viz", action="store_true",
                        help="Skip feature visualization step")
    parser.add_argument("--feature-viz-nb-top-concepts", type=int, default=4,
                        help="Number of top concepts to visualize (default: 4)")
    parser.add_argument("--accuracy", type=float, default=None,
                        help="Accuracy threshold for filtering boxes (overrides model defaults)")
    args = parser.parse_args()

    # Validate framework-specific constraints
    if args.framework == "tf":
        if args.model != "retinanet":
            parser.error("TensorFlow framework only supports 'retinanet' model")
        if not args.skip_feature_viz:
            print("WARNING: Feature visualization not available for TensorFlow, skipping...")
            args.skip_feature_viz = True

    return args


def load_model(model_name, config, device, framework="torch"):
    if framework == "tf":
        # TensorFlow models
        if model_name == "retinanet":
            from keras_cv.models import RetinaNet
            model = RetinaNet.from_preset("retinanet_resnet50_pascalvoc")
            return model
    else:
        # PyTorch models
        if model_name == "detr":
            model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
            model.eval()
            return model

        elif model_name == "retinanet":
            from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            model = retinanet_resnet50_fpn_v2(weights=weights)
            model.eval()
            model = model.to(device)
            return model, weights

        elif model_name == "fasterrcnn":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            model.eval()
            model = model.to(device)
            return model, weights

        elif model_name == "fcos":
            from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
            weights = FCOS_ResNet50_FPN_Weights.DEFAULT
            model = fcos_resnet50_fpn(weights=weights, box_score_thresh=0.9)
            model.eval()
            model = model.to(device)
            return model, weights

        elif model_name == "ssd":
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
            weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            model = ssdlite320_mobilenet_v3_large(weights=weights)
            model.eval()
            model = model.to(device)
            return model, weights

        elif model_name == "yolo":
            from ultralytics import YOLO
            model = YOLO(config["model_path"]).to(device)
            model.eval()
            return model

        elif model_name == "yolo_relu":
            from ultralytics import YOLO
            model = YOLO(config["model_path"]).to(device)
            model.eval()
            return model


def build_latent_extractor(model_name, model, config, device, framework="torch"):
    if framework == "tf":
        # TensorFlow extractors
        if model_name == "retinanet":
            from xplique.concepts.tf.latent_data_retinanet import RetinaNetExtractorBuilder
            nb_classes = len(PASCAL_VOC_CLASSES)
            return RetinaNetExtractorBuilder.build(model, nb_classes, index_activations=-1, batch_size=8)
    else:
        # PyTorch extractors
        if model_name == "detr":
            from xplique.concepts.torch.latent_data_detr import DetrExtractorBuilder
            return DetrExtractorBuilder.build(model=model, batch_size=config["batch_size"], device=device)

        elif model_name == "retinanet":
            from xplique.concepts.torch.latent_data_retinanet import RetinanetExtractorBuilder
            nb_classes = len(COCO_CLASSES)
            return RetinanetExtractorBuilder.build(model, device=device,
                                                   nb_classes=nb_classes,
                                                   extraction_layer=config["extraction_layer"])

        elif model_name == "fasterrcnn":
            from xplique.concepts.torch.latent_data_faster_rcnn import FasterRcnnExtractorBuilder
            nb_classes = len(COCO_CLASSES)
            return FasterRcnnExtractorBuilder.build(model, device=device,
                                                    nb_classes=nb_classes,
                                                    extraction_layer=config["extraction_layer"],
                                                    extract_location=args.extract_location)

        elif model_name == "fcos":
            from xplique.concepts.torch.latent_data_fcos import FcosExtractorBuilder
            return FcosExtractorBuilder.build(model, device=device)

        elif model_name == "ssd":
            from xplique.concepts.torch.latent_data_ssd import SSDExtractorBuilder
            nb_classes = len(COCO_CLASSES)
            return SSDExtractorBuilder.build(model, device=device,
                                             nb_classes=nb_classes,
                                             extraction_layer=config["extraction_layer"])

        elif model_name == "yolo":
            from xplique.concepts.torch.latent_data_yolo import YoloExtractorBuilder
            detection_model = model.model
            return YoloExtractorBuilder.build(detection_model,
                                              extraction_layer=config["extraction_layer"],
                                              batch_size=config["batch_size"])

        elif model_name == "yolo_relu":
            from xplique.concepts.torch.latent_data_yolo import YoloExtractorBuilder
            detection_model = model.model
            return YoloExtractorBuilder.build(detection_model,
                                              extraction_layer=config["extraction_layer"],
                                              batch_size=config["batch_size"])


def preprocess_images(raw_images, model_name, config, weights=None, framework="torch"):
    if framework == "tf":
        # TensorFlow preprocessing
        import tensorflow as tf
        images_size = (640, 640)
        input_arrays = []
        for img in raw_images:
            img_resized = img.resize(images_size)
            img_np = np.array(img_resized, dtype=np.float32)
            input_arrays.append(img_np)
        input_tensor = tf.stack(input_arrays, axis=0)
        return input_tensor
    else:
        # PyTorch preprocessing
        if config["use_weights_transform"]:
            preprocess = weights.transforms()
            input_tensors = [img.resize(config["image_size"]) for img in raw_images]
            input_tensors = [preprocess(img) for img in input_tensors]
            input_tensor = torch.stack(input_tensors, dim=0)
        elif model_name in ["yolo", "yolo_relu"]:
            transform = T.Compose([
                T.Resize(config["image_size"]),
                T.ToTensor()
            ])
            input_tensors = [transform(img).unsqueeze(0) for img in raw_images]
            input_tensor = torch.cat(input_tensors, dim=0)
        else:
            transform = T.Compose([
                T.Resize(config["image_size"]),
                T.ToTensor(),
                T.Normalize(config["custom_normalize"][0], config["custom_normalize"][1])
            ])
            input_tensors = [transform(img).unsqueeze(0) for img in raw_images]
            input_tensor = torch.cat(input_tensors, dim=0)

        return input_tensor


def get_class_names(model_name, model, framework="torch"):
    if framework == "tf":
        # TensorFlow RetinaNet uses PASCAL VOC classes
        return PASCAL_VOC_CLASSES
    else:
        # PyTorch models
        if model_name in ["yolo", "yolo_relu"]:
            return list(model.names.values())
        else:
            return COCO_CLASSES


def create_factorizer(nmf_type, num_concepts, device):
    if nmf_type == "sklearn":
        return None
    elif nmf_type == "oc_semi_nmf":
        from overcomplete.optimization import SemiNMF
        from xplique.concepts.torch.factorizer import OvercompleteFactorizer
        return OvercompleteFactorizer(
            optimizer_class=SemiNMF,
            nb_concepts=num_concepts,
            device=device
        )


def plot_concepts_feature_viz(craft, input_tensor, indices_to_plot, device, model_input_size, output_dir, suffix="", file_number=10):
    def objective(images, concept_id=0):
        latent_data_and_coeffs_u = craft.encode(images, differentiable=True)
        coeffs_u_list = [pair.coeffs_u for pair in latent_data_and_coeffs_u]
        coeffs_u = torch.cat(coeffs_u_list)
        return torch.mean(coeffs_u[:, :, :, concept_id])

    def objective_for_concept(concept_id):
        return partial(objective, concept_id=concept_id)

    n = len(indices_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

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
    args = parse_args()

    config = MODEL_CONFIGS[args.model]
    
    # Override config for TensorFlow framework
    if args.framework == "tf" and args.model == "retinanet":
        # TF RetinaNet uses normalized boxes
        config = config.copy()
        config["normalized"] = True
    
    num_images = args.num_images if args.num_images is not None else config["default_num_images"]
    start_offset = args.start_offset if args.start_offset is not None else config["default_start_offset"]

    # Override extraction_layer if provided via CLI
    if args.extraction_layer is not None and "extraction_layer" in config:
        config["extraction_layer"] = args.extraction_layer

    # Override accuracy values if provided via CLI
    if args.accuracy is not None:
        config["accuracy"] = args.accuracy

    nmf_suffix = "" if args.nmf_type == "sklearn" else f"_{args.nmf_type}"
    accuracy_suffix = "" if args.accuracy is None else f"_acc{args.accuracy}"
    extract_location_suffix = "" if args.extract_location == "resnet" else f"_{args.extract_location}"
    output_dir_name = f"{args.framework}_{args.model}_craft{nmf_suffix}{extract_location_suffix}_{args.class_name}{accuracy_suffix}"

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, output_dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device handling: PyTorch uses device, TensorFlow doesn't need it
    if args.framework == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        device = None
        print("Using TensorFlow framework")

    print("\n" + "="*80)
    print("1. Loading Images with Specified Class")
    print("="*80)

    COCO_ROOT = os.path.expanduser('~/data/coco')
    IMAGE_DIR = os.path.join(COCO_ROOT, 'val2017')
    ANNOTATION_FILE = os.path.join(COCO_ROOT, 'annotations', 'instances_val2017.json')

    print("Loading COCO annotations...")
    coco = COCO(ANNOTATION_FILE)

    cat_ids = coco.getCatIds(catNms=[args.class_name])
    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"Found {len(img_ids)} images containing '{args.class_name}' class")

    selected_img_ids = img_ids[start_offset:start_offset + num_images]
    print(f"Selected {len(selected_img_ids)} images for processing (offset: {start_offset})")

    raw_images = []
    for img_id in selected_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
        raw_images.append(Image.open(img_path))

    print("\n" + "="*80)
    print("2. Displaying All Loaded Images")
    print("="*80)

    num_display = min(len(raw_images), 20)
    cols = 4
    rows = (num_display + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if num_display > 1 else [axes]

    for idx in range(num_display):
        axes[idx].imshow(raw_images[idx])
        axes[idx].set_title(f'Image {idx}')
        axes[idx].axis('off')

    for idx in range(num_display, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "01_loaded_images.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print(f"3. Loading {args.model.upper()} Model")
    print("="*80)

    model_result = load_model(args.model, config, device, framework=args.framework)
    if isinstance(model_result, tuple):
        model, weights = model_result
    else:
        model = model_result
        weights = None

    print(f"{args.model.upper()} model loaded successfully")

    classes_names = get_class_names(args.model, model, framework=args.framework)

    label_to_color = {
        f'{args.class_name}': 'r',
        'bicycle': 'b',
        'car': 'g',
        'motorcycle': 'y',
        'truck': 'orange'
    }

    print("\n" + "="*80)
    print("Preprocessing Images")
    print("="*80)

    input_tensor = preprocess_images(raw_images, args.model, config, weights, framework=args.framework)
    if args.framework == "torch":
        input_tensor = input_tensor.to(device)

    print(f"Batched input tensor shape: {input_tensor.shape}")
    print(f"Number of images: {len(raw_images)}")

    print("\n" + "="*80)
    print("4. Loading Latent Extractor")
    print("="*80)

    latent_extractor = build_latent_extractor(args.model, model, config, device, framework=args.framework)
    print("Latent extractor built successfully")

    print("\n" + "="*80)
    print("5. Displaying Boxes for First Image")
    print("="*80)

    first_image_tensor = input_tensor[0:1]
    results = latent_extractor(first_image_tensor)
    
    # Select appropriate BoxManager based on framework
    if args.framework == "tf":
        box_manager = TfBoxManager(config["box_format"], normalized=config["normalized"])
    else:
        box_manager = TorchBoxManager(config["box_format"], normalized=config["normalized"])

    box_to_display = results[0].filter(accuracy=config["accuracy"],
                                       class_id=classes_names.index(args.class_name))

    display_image = raw_images[0].resize(config["image_size"]) if config["normalized"] == False else raw_images[0]
    fig = display_image_with_boxes(display_image, box_to_display, box_manager, classes_names, label_to_color)

    output_path = os.path.join(OUTPUT_DIR, "02_first_image_boxes.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print("6. Initializing and Running CRAFT on All Images")
    print("="*80)

    factorizer = create_factorizer(args.nmf_type, args.num_concepts, device)

    # Select appropriate CRAFT class based on framework
    if args.framework == "tf":
        craft = CraftTf(latent_extractor=latent_extractor,
                        number_of_concepts=args.num_concepts,
                        factorizer=factorizer)
    else:
        craft = CraftTorch(latent_extractor=latent_extractor,
                           number_of_concepts=args.num_concepts,
                           device=str(device),
                           factorizer=factorizer)
    _ = craft.fit(input_tensor)
    print("CRAFT fitting completed")

    print("\n" + "="*80)
    print("7. Displaying Concepts")
    print("="*80)

    fig = craft.display_images_per_concept(input_tensor[:10], order=None, filter_percentile=80, clip_percentile=5)

    output_path = os.path.join(OUTPUT_DIR, "03_concepts_unordered.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print(f"8. Ranking Concepts by Importance for '{args.class_name}' class")
    print("="*80)

    operator = xplique.Tasks.OBJECT_DETECTION
    class_id = classes_names.index(f"{args.class_name}")

    # Evaluate with Gradient Input
    importances_gi = craft.estimate_importance(input_tensor[:20], operator, class_id,
                                               accuracy=config["accuracy"],
                                               method="gradient_input")
    order_gi = importances_gi.argsort()[::-1]

    print(f"Concept importances (Gradient Input) for '{args.class_name}': {importances_gi}")

    # Conditionally evaluate with Sobol
    run_sobol = args.num_images_sobol > 0
    if run_sobol:
        print(f"Running Sobol attribution with {args.num_images_sobol} images...")
        importances_sobol = craft.estimate_importance(input_tensor[:args.num_images_sobol], operator, class_id,
                                                       accuracy=config["accuracy"],
                                                       method="sobol")
        order_sobol = importances_sobol.argsort()[::-1]

        print(f"Concept importances (Sobol) for '{args.class_name}': {importances_sobol}")
    else:
        print("Skipping Sobol attribution (num_images_sobol=0)")

    # Display concepts ordered by Gradient Input
    fig = craft.display_images_per_concept(input_tensor[:5], order=order_gi, filter_percentile=80, clip_percentile=5)

    output_path = os.path.join(OUTPUT_DIR, "04_concepts_ordered_gi.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Display concepts ordered by Sobol (if run)
    if run_sobol:
        fig = craft.display_images_per_concept(input_tensor[:5], order=order_sobol, filter_percentile=80, clip_percentile=5)

        output_path = os.path.join(OUTPUT_DIR, "05_concepts_ordered_sobol.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    # Plot importance bars for Gradient Input
    plt.figure(figsize=(10, 6))
    plt.bar(list(range(len(importances_gi))), importances_gi)
    plt.xlabel('Concept ID')
    plt.ylabel('Importance')
    plt.title(f'Concept Importances (Gradient Input) for "{args.class_name}" class')

    output_path = os.path.join(OUTPUT_DIR, "06_importance_bars_gi.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Plot importance bars for Sobol (if run)
    if run_sobol:
        plt.figure(figsize=(10, 6))
        plt.bar(list(range(len(importances_sobol))), importances_sobol)
        plt.xlabel('Concept ID')
        plt.ylabel('Importance')
        plt.title(f'Concept Importances (Sobol) for "{args.class_name}" class')

        output_path = os.path.join(OUTPUT_DIR, "07_importance_bars_sobol.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    print("\n" + "="*80)
    print("9. Displaying Top Images per Concept")
    print("="*80)

    # Use Gradient Input order for top images
    # Generate top images for Gradient Input
    fig = craft.display_top_images_per_concept(
        input_tensor,
        nb_top_images=5,
        filter_percentile=80,
        clip_percentile=5,
        order=order_gi
    )

    output_path = os.path.join(OUTPUT_DIR, "08_top_images_per_concept_gi.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Generate top images for Sobol (if run)
    if run_sobol:
        fig = craft.display_top_images_per_concept(
            input_tensor,
            nb_top_images=5,
            filter_percentile=80,
            clip_percentile=5,
            order=order_sobol
        )

        output_path = os.path.join(OUTPUT_DIR, "09_top_images_per_concept_sobol.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    print("\n" + "="*80)
    print("10. Feature Visualization")
    print("="*80)

    if args.skip_feature_viz:
        print("Skipping feature visualization (--skip-feature-viz flag set)")
    else:
        model_input_size = config["image_size"][0]
        nb_top_concepts = args.feature_viz_nb_top_concepts

        # Feature viz for Gradient Input
        plot_concepts_feature_viz(craft, input_tensor, order_gi[:nb_top_concepts], device, model_input_size, OUTPUT_DIR, suffix="gi", file_number=10)

        # Feature viz for Sobol (if run)
        if run_sobol:
            plot_concepts_feature_viz(craft, input_tensor, order_sobol[:nb_top_concepts], device, model_input_size, OUTPUT_DIR, suffix="sobol", file_number=11)

    print("\n" + "="*80)
    print("All outputs saved to:", OUTPUT_DIR)
    print("="*80)


if __name__ == "__main__":
    main()
