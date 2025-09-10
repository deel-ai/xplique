import os
import pprint

import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras_cv.models import RetinaNet

import xplique
from xplique.attributions import Saliency
from xplique.plots import plot_attributions
from xplique.utils_functions.object_detection.tf.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.common.box_manager import BoxFormat
from xplique.utils_functions.object_detection.tf.box_manager import filter_boxes as filter_boxes_tf
from xplique.concepts.latent_data_retinanet import buildTfRetinaNetLatentExtractor
from xplique.concepts import HolisticCraftObjectDetectionTf as Craft

from xplique.plots.display_image_with_boxes import display_image_with_boxes_tf


pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(tf.__version__) # Check the version of tensorflow used

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactive complètement le GPU

# Ou alternativement, forcer TensorFlow à utiliser le CPU
tf.config.set_visible_devices([], 'GPU')

test_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="module")
def image_data():
    # Load and preprocess the image
    raw_image = Image.open("img.jpg")
    def resize_image(image, target_height=640):
        width, height = image.size
        target_width = int(width * target_height / height)
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    resized_image = resize_image(raw_image)
    image = resized_image.crop((100, 0, 740, 640))
   
    img_np = np.array(image, dtype=np.float32)
    input_tensor = tf.expand_dims(img_np, axis=0)
    return image, input_tensor

def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 640)
    assert image.size == expected_size

@pytest.fixture(scope="module")
def model_data(image_data):
    _, input_tensor = image_data
    model = RetinaNet.from_preset("retinanet_resnet50_pascalvoc")
    processed_results = model.predict(input_tensor)
    return model, processed_results

def test_model_outputs(model_data):
    _, processed_results = model_data
    assert list(processed_results.keys()) == ['boxes', 'confidence', 'classes', 'num_detections']

def test_gradients_predict(image_data, model_data):
    image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model.predict, input_tensor)
    assert check is False, "Model gradients should not be computed successfully in normal / predict mode."

def test_gradients_model_call(image_data, model_data):
    image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check, "Model gradients should be computed successfully when calling the model."



@pytest.fixture(scope="module")
def dataset_classes():
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
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

    latent_extractor = buildTfRetinaNetLatentExtractor(model, nb_classes)
    return latent_extractor

def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data Retinanet:", results)
    assert results.shape == [1, 76725, 25], "Latent data shape should be [1, 76725, 25]."


    fig = display_image_with_boxes_tf(image, results[0], BoxFormat.XYXY, True, classes_names, label_to_color, accuracy=0.85)

    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/retinanet_tf_latent_extractor_output.png")

def test_latent_extractor_gradients(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."

def test_latent_extractor_saliency(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data
    

    operator=xplique.Tasks.OBJECT_DETECTION
    targets = latent_extractor(input_tensor)
    filtered_targets = filter_boxes_tf(targets, accuracy=0.9, class_id=classes_names.index('car'))
    box_to_explain = np.array(filtered_targets)

    explainer = Saliency(latent_extractor, operator=operator, batch_size=None)
    explanation = explainer.explain(input_tensor, targets=box_to_explain)
 
    plot_attributions(explanation, [np.array(image)], img_size=6.,
                    cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.5)
    # save to file in the directory of the test script is located
    fig = plt.gcf()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/retinanet_tf_latent_extractor_saliency.png")


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
    assert coeffs_u.shape == (1, 20, 20, 10), "Latent data shape should be (1, 20, 20, 10)."

    result = craft.decode(latent_data, coeffs_u)
    print(result.shape)
    assert result.shape == (1, 76725, 25), "Decoded result shape should be (1, 76725, 25)."

    display_image_with_boxes_tf(image, result[0], BoxFormat.XYXY, True, classes_names, label_to_color, accuracy=0.85)
    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f"{test_dir}/retinanet_tf_latent_extractor_craft_reencode.png")

def test_craft_concepts(image_data, craft_data):
    image, input_tensor = image_data
    craft = craft_data
    fig = craft.display_images_per_concept(input_tensor, order=None, filter_percentile=80, clip_percentile=5)
    # save to file in the directory of the test script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{test_dir}/retinanet_tf_latent_extractor_concepts_unsorted.png")

def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    operator = xplique.Tasks.OBJECT_DETECTION
    class_id = classes_names.index("person")
    importances_gi = craft.estimate_importance_gradient_input_xplique(input_tensor, operator, class_id, accuracy=0.6, batch_size=1)

    order = importances_gi.argsort()[::-1]
    fig = craft.display_images_per_concept(input_tensor, order=order, filter_percentile=80, clip_percentile=5)
    
    fig.savefig(f"{test_dir}/retinanet_tf_latent_extractor_concepts_sorted.png")
