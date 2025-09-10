import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import matplotlib.patches as mpatches

from xplique.utils_functions.object_detection.common.box_manager import BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager
from xplique.utils_functions.object_detection.torch.box_formatter import NBCTensor
from xplique.utils_functions.object_detection.tf.box_manager import TfBoxManager


def display_image_with_boxes(image, nbc_annotations: list, box_format: BoxFormat, box_is_normalized: bool, classes_labels: list[str], label_to_color: dict, accuracy: float=0.55):
    
    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    fig, ax = plt.subplots()
    ax.imshow(image)

    box_manager = TorchBoxManager(box_format, normalized=box_is_normalized)
    keep = nbc_annotations[:,4] > accuracy
    # print("Number of kept annotations:", keep.sum().item())
    xyxy_boxes = nbc_annotations[keep, :4]
    scores = nbc_annotations[keep, 4]
    probas = nbc_annotations[keep, 5:]
    # rescaled_boxes = xyxy_boxes 
    rescaled_boxes = box_manager.resize(xyxy_boxes, (1, 1), image.size)

    found_labels = set()
    for box_coords, score, proba in zip(rescaled_boxes, scores, probas):
        if box_is_normalized:
            # rescale the boxes to the original image size
            xmin, ymin, xmax, ymax = box_manager.rescale_bboxes(box_coords, image.size)
        else:
            # box not normalized, assume they are already in the correct image dimensions
            xmin, ymin, xmax, ymax = box_coords
        xmin, ymin, xmax, ymax = xmin.detach().cpu().numpy(), ymin.detach().cpu().numpy(), xmax.detach().cpu().numpy(), ymax.detach().cpu().numpy()
        cl = proba.argmax().item()
        color = label_to_color.get(classes_labels[cl])
        if color is None:
            print(f"Warning: No color defined for class '{classes_labels[cl]}'. Using default color 'black'.")
        name = class_id_to_label.get(cl, 'unknown')
        found_labels.add(name)
        # print(f"cl:{cl}, Drawing box for {name} with color {color} at coords ({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        ax.text(xmin, ymin-15, f"{score:.2f}", color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.0))
        

    handles = [mpatches.Patch(color=color, label=label)
            for label, color in label_to_color.items() if label in found_labels]
    plt.legend(handles=handles)
    return fig


def display_image_with_boxes_nbctensor(image, nbc_annotations: NBCTensor, box_format: BoxFormat, box_is_normalized: bool, classes_labels: list[str], label_to_color: dict):
    
    if nbc_annotations.shape[0] != 1:
        raise ValueError(f"nbc_annotations should have a batch size of 1 (for 1 image), got {nbc_annotations.shape[0]}")

    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    fig, ax = plt.subplots()
    ax.imshow(image)

    box_manager = TorchBoxManager(box_format, normalized=box_is_normalized)
    boxes = nbc_annotations.boxes(0)
    scores = nbc_annotations.scores(0)
    probas = nbc_annotations.probas(0)

    rescaled_boxes = box_manager.resize(boxes, (1, 1), image.size)

    found_labels = set()
    for box_coords, score, proba in zip(rescaled_boxes, scores, probas):
        if box_is_normalized:
            # rescale the boxes to the original image size
            xmin, ymin, xmax, ymax = box_manager.rescale_bboxes(box_coords, image.size)
        else:
            # box not normalized, assume they are already in the correct image dimensions
            xmin, ymin, xmax, ymax = box_coords
        xmin, ymin, xmax, ymax = xmin.detach().cpu().numpy(), ymin.detach().cpu().numpy(), xmax.detach().cpu().numpy(), ymax.detach().cpu().numpy()
        cl = proba.argmax().item()
        color = label_to_color.get(classes_labels[cl])
        if color is None:
            print(f"Warning: No color defined for class '{classes_labels[cl]}'. Using default color 'black'.")
        name = class_id_to_label.get(cl, 'unknown')
        found_labels.add(name)
        # print(f"cl:{cl}, Drawing box for {name} with color {color} at coords ({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        ax.text(xmin, ymin-15, f"{score:.2f}", color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.0))
        

    handles = [mpatches.Patch(color=color, label=label)
            for label, color in label_to_color.items() if label in found_labels]
    plt.legend(handles=handles)
    return fig

def display_image_with_boxes_tf(image, nbc_annotations: tf.Tensor, box_format: BoxFormat, box_is_normalized: bool, classes_labels: list[str], label_to_color: dict, accuracy: float=0.55):
    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    fig, ax = plt.subplots()
    ax.imshow(image)

    box_manager = TfBoxManager(box_format, normalized=box_is_normalized)
    nbc_annotations = nbc_annotations.numpy()
    keep = nbc_annotations[:,4] > accuracy
    print("Number of kept annotations:", keep.sum().item())
    xyxy_boxes = nbc_annotations[keep, :4]
    
    scores = nbc_annotations[keep, 4]
    probas = nbc_annotations[keep, 5:]
    # rescaled_boxes = xyxy_boxes 
    rescaled_boxes = box_manager.resize(xyxy_boxes, (1, 1), image.size)

    found_labels = set()
    for box_coords, score, proba in zip(rescaled_boxes, scores, probas):
        if box_is_normalized:
            # rescale the boxes to the original image size
            xmin, ymin, xmax, ymax = box_manager.rescale_bboxes(box_coords, image.size)
        else:
            # box not normalized, assume they are already in the correct image dimensions
            xmin, ymin, xmax, ymax = box_coords
        
        cl = proba.argmax().item()
        color = label_to_color.get(classes_labels[cl])
        if color is None:
            print(f"Warning: No color defined for class '{classes_labels[cl]}'. Using default color 'black'.")
        name = class_id_to_label.get(cl, 'unknown')
        found_labels.add(name)
        print(f"cl:{cl}, Drawing box for {name} with color {color} at coords ({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        ax.text(xmin, ymin-15, f"{score:.2f}", color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.0))
        

    handles = [mpatches.Patch(color=color, label=label)
            for label, color in label_to_color.items() if label in found_labels]
    plt.legend(handles=handles)
    return fig
