import os
import pprint

import pytest
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import xplique
from xplique.attributions import Saliency
from xplique.plots import plot_attributions
from xplique.utils_functions.object_detection.torch.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.common.box_manager import BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import filter_boxes
from xplique.concepts.latent_data_detr import LatentDataDetr, buildTorchDetrLatentExtractor
from xplique.concepts import HolisticCraftObjectDetectionTorch as Craft

from xplique.plots.display_image_with_boxes import display_image_with_boxes
from xplique.wrappers import TorchWrapper

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(torch.__version__) # Check the version of torch used

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
test_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@pytest.fixture(scope="module")
def image_data():
    # Load and preprocess the image
    raw_image = Image.open("img.jpg")

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # preprocess and batch the image
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    return raw_image, input_tensor

def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="module")
def model_data(image_data):
    _, input_tensor = image_data
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
    model.eval()
    processed_results = model(input_tensor)
    return model, processed_results

def test_model_outputs(model_data):
    _, processed_results = model_data
    assert list(processed_results.keys()) == ['pred_logits', 'pred_boxes']

def test_gradients_model_original(image_data, model_data):
    image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check == False, "Model gradients should be not computed successfully when calling the default Detr model."

@pytest.fixture(scope="module")
def dataset_classes():
    # COCO classes
    CLASSES = [
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
    nb_classes = len(CLASSES)
    label_to_color = {'person': 'r',
                'bicycle': 'b',
                'car': 'g',
                'motorcycle': 'y',
                'truck': 'orange'}
    return CLASSES, nb_classes, label_to_color

@pytest.fixture(scope="module")
def latent_extractor_data(dataset_classes, model_data):
    classes_names, nb_classes, label_to_color = dataset_classes
    model, _ = model_data

    latent_extractor = buildTorchDetrLatentExtractor(model)
    return latent_extractor

def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data Detr:", results)
    assert results.shape == torch.Size([1, 100, 96]), "Latent data shape should be [1, 100, 96]."

    fig = display_image_with_boxes(image, results[0], BoxFormat.XYXY, True, classes_names, label_to_color, accuracy=0.85)

    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/detr_torch_latent_extractor_output.png")

def test_latent_extractor_gradients(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."

def test_latent_extractor_saliency(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    targets = latent_extractor(input_tensor)
    filtered_targets = filter_boxes(targets, accuracy=0.9, class_id=classes_names.index('car'))
    box_to_explain = np.array([target.detach().cpu().numpy() for target in filtered_targets])

    # @TODO: filter box on targets coming from the torch wrapped model -> filter_boxes_tf ?
    torch_wrapped_model = TorchWrapper(latent_extractor, device=device, is_channel_first=True)
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

    explainer = Saliency(torch_wrapped_model, operator=xplique.Tasks.OBJECT_DETECTION, batch_size=1)
    explanation = explainer.explain(input_tensor_tf_dim, targets=box_to_explain)
 
    plot_attributions(explanation, [np.array(image)], img_size=6.,
                    cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.5)
    # save to file in the directory of the test script is located
    fig = plt.gcf()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/detr_torch_latent_extractor_saliency.png")

@pytest.fixture(scope="module")
def craft_data(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    craft = Craft(latent_extractor = latent_extractor,
                number_of_concepts = 10)
    craft.fit(input_tensor)
    return craft

def test_craft_reencode(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    latent_data, coeffs_u = craft.encode(input_tensor)
    print(coeffs_u.shape)
    assert coeffs_u.shape == (1, 25, 25, 10), "Latent data shape should be (1, 25, 25, 10)."

    result = craft.decode(latent_data, coeffs_u)
    print(result.shape)
    assert result.shape == torch.Size([1, 100, 96]), "Decoded result shape should be (1, 100, 96)."

    display_image_with_boxes(image, result[0], BoxFormat.XYXY, True, classes_names, label_to_color, accuracy=0.85)
    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f"{test_dir}/detr_torch_latent_extractor_craft_reencode.png")

def test_craft_concepts(image_data, craft_data):
    image, input_tensor = image_data
    craft = craft_data
    fig = craft.display_images_per_concept(input_tensor, order=None, filter_percentile=80, clip_percentile=5)
    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/detr_torch_latent_extractor_concepts_unsorted.png")

def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    operator = xplique.Tasks.OBJECT_DETECTION
    class_id = classes_names.index("person")
    importances_gi = craft.estimate_importance_gradient_input_xplique(input_tensor, operator, class_id, accuracy=0.6, batch_size=1)

    order = importances_gi.argsort()[::-1]
    fig = craft.display_images_per_concept(input_tensor, order=order, filter_percentile=80, clip_percentile=5)
    
    fig.savefig(f"{test_dir}/detr_torch_latent_extractor_concepts_sorted.png")
