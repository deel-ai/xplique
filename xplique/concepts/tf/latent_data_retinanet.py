import types
from typing import Union

import numpy as np
import tensorflow as tf
from keras_cv.src import bounding_box
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes

from xplique.utils_functions.object_detection.tf.box_formatter import (
    RetinaNetProcessedBoxFormatter,
)

from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TfLatentExtractor


class TfLatentDataRetinanet(LatentData):
    """
    Stores latent representations (feature maps) from RetinaNet's ResNet backbone.

    This class encapsulates the multi-scale feature pyramid outputs from the ResNet
    backbone of a RetinaNet model. Features are typically stored as a dictionary with
    keys like 'P3', 'P4', 'P5' representing different pyramid levels.

    Attributes
    ----------
    resnet_features
        Dictionary of feature maps from ResNet backbone. Keys are pyramid levels
        (e.g., 'P3', 'P4', 'P5') and values are tensors with shapes like:
        - P3: (batch, 80, 80, 512)
        - P4: (batch, 40, 40, 1024)
        - P5: (batch, 20, 20, 2048)
    image_shape
        Tuple representing the shape of the input image (height, width, channels).
    index_activations
        Index specifying which feature map to use as activations. Default is -1 (last feature).
    """
    index_activations = -1
    resnet_features: dict
    # typically a dict of resnet features with keys like 'P3', 'P4', 'P5'
    # Feature P3: np.shape (1, 80, 80, 512)
    # Feature P4: shape (1, 40, 40, 1024)
    # Feature P5: shape (1, 20, 20, 2048)
    image_shape: tuple

    def __init__(
            self,
            resnet_features: dict,
            image_shape: tuple,
            index_activations: int = -1) -> None:
        """
        Initialize RetinaNet latent data with feature maps and image shape.

        Parameters
        ----------
        resnet_features
            Dictionary of feature maps from ResNet backbone.
        image_shape
            Tuple representing the input image shape.
        index_activations
            Index specifying which feature map to use. Default is -1.
        """
        self.resnet_features = resnet_features
        self.image_shape = image_shape
        self.index_activations = index_activations

    def __len__(self) -> int:
        """
        Return the batch size from the feature maps.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        key = list(self.resnet_features.keys())[self.index_activations]
        return self.resnet_features[key].shape[0]

    def print_infos(self) -> None:
        """
        Print information about stored feature maps and image shape.

        Displays the keys and shapes of all feature maps in the ResNet features
        dictionary, as well as the input image shape.
        """
        print("LatentDataRetinanet:")
        for key, feature in self.resnet_features.items():
            print(f"\tFeature {key}: shape {feature.shape}")
        print(f"Image shape: {self.image_shape}")

    def get_activations(
            self,
            as_numpy: bool = True,
            keep_gradients: bool = False) -> Union[np.ndarray, tf.Tensor]:
        """
        Extract the feature map at the specified index.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information (not used in this implementation).

        Returns
        -------
        activations
            Feature map as numpy array or tensor, depending on as_numpy parameter.
        """
        key = list(self.resnet_features.keys())[self.index_activations]
        activations = self.resnet_features[key]

        if as_numpy:
            if isinstance(activations, tf.Tensor):
                activations = activations.numpy()

        return activations

    def check_features_positive(self) -> None:
        """
        Verify that all feature maps contain only non-negative values.

        Raises
        ------
        ValueError
            If any feature map contains negative values.
        """
        for feature in self.resnet_features:
            if tf.reduce_any(feature < 0):
                raise ValueError("Features contain negative values, which is unexpected.")

    def set_activations(self, values: Union[tf.Tensor, np.ndarray]) -> None:
        """
        Update the feature map at the specified index.

        Parameters
        ----------
        values
            New feature map values as tf.Tensor or np.ndarray.

        Raises
        ------
        TypeError
            If values is not a tf.Tensor or np.ndarray.
        """
        key = list(self.resnet_features.keys())[self.index_activations]
        if isinstance(values, tf.Tensor):
            self.resnet_features[key] = values
        elif isinstance(values, np.ndarray):
            self.resnet_features[key] = tf.convert_to_tensor(values)
        else:
            raise TypeError(f"Unsupported type: {type(values)}. Expected tf.Tensor or np.ndarray")


class RetinaNetExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for RetinaNet models.

    This class provides methods to construct a TfLatentExtractor specifically
    configured for RetinaNet object detection models. It defines the forward
    pass split into feature extraction (g) and prediction decoding (h).
    """

    @classmethod
    def build(cls, model, nb_classes, index_activations=-1,
              batch_size: int = 1) -> 'TfLatentExtractor':
        """
        Build a LatentExtractor for a RetinaNet model.

        This method creates custom g and h functions that split the model's forward
        pass: g extracts backbone features, and h processes them through the detection
        head and decodes predictions.

        Parameters
        ----------
        model
            TensorFlow RetinaNet model instance with feature_extractor, feature_pyramid,
            classification_head, and box_head attributes.
        nb_classes
            Number of object classes the model detects.
        index_activations
            Index specifying which feature pyramid level to use as activations. Default is -1.
        batch_size
            Batch size for processing. Default is 1.
        Returns
        -------
        latent_extractor
            Configured TfLatentExtractor instance for the RetinaNet model.
        """

        def g(self, samples: tf.Tensor) -> TfLatentDataRetinanet:
            backbone_outputs = self.feature_extractor(samples, training=False)
            return TfLatentDataRetinanet(backbone_outputs, tuple(
                samples[0].shape), index_activations=index_activations)

        def h(self, latent_data: TfLatentDataRetinanet) -> dict:
            backbone_outputs, image_shape = latent_data.resnet_features, latent_data.image_shape
            features = self.feature_pyramid(backbone_outputs, training=False)
            cls_outputs = []
            box_outputs = []
            for feature in features:
                cls_outputs.append(self.classification_head(feature))
                box_outputs.append(self.box_head(feature))

            # 4. Concatener les sorties
            cls_outputs = tf.concat(
                [tf.reshape(c, [tf.shape(c)[0], -1, self.num_classes]) for c in cls_outputs], axis=1)
            box_outputs = tf.concat([tf.reshape(b, [tf.shape(b)[0], -1, 4])
                                    for b in box_outputs], axis=1)

            # 5. Si training=False, appliquer le post-processing
            # if not training:
            def decode_predictions_reworked(predictions, image_shape):
                BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]
                box_pred, cls_pred = predictions["box"], predictions["classification"]
                # box_pred is on "center_yxhw" format, convert to target format.
                # image_shape = tuple(images[0].shape)
                anchors = model.anchor_generator(image_shape=image_shape)
                anchors = ops.concatenate([a for a in anchors.values()], axis=0)

                box_pred = _decode_deltas_to_boxes(
                    anchors=anchors,
                    boxes_delta=box_pred,
                    anchor_format=model.anchor_generator.bounding_box_format,
                    box_format=model.bounding_box_format,
                    variance=BOX_VARIANCE,
                    image_shape=image_shape,
                )
                # box_pred is now in "self.bounding_box_format" format
                box_pred = bounding_box.convert_format(
                    box_pred,
                    source=model.bounding_box_format,
                    target=model.prediction_decoder.bounding_box_format,
                    image_shape=image_shape,
                )

                # fonction a remplacer: model.prediction_decoder()
                # (self, box_prediction, class_prediction, images=None, image_shape=None)
                box_prediction = box_pred
                class_prediction = cls_pred

                # Logits to probas
                predictions = ops.sigmoid(class_prediction)
                # predicted_class = ops.argmax(predictions, axis=-1)
                # Take the class with the highest confidence
                confidence = ops.max(predictions, axis=-1)
                classes = ops.argmax(predictions, axis=-1)

                bounding_boxes = {
                    "boxes": box_prediction,
                    "scores": predictions,
                    "confidence": confidence,
                    "classes": classes,
                }
                return bounding_boxes

            return decode_predictions_reworked(
                {"classification": cls_outputs, "box": box_outputs},
                image_shape
            )

        model.h = types.MethodType(h, model)
        model.g = types.MethodType(g, model)

        processed_formatter = RetinaNetProcessedBoxFormatter(
            nb_classes=nb_classes, input_image_size=(
                640, 640), output_image_size=(
                640, 640))
        latent_extractor = TfLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=TfLatentDataRetinanet,
            output_formatter=processed_formatter,
            batch_size=batch_size)
        return latent_extractor
