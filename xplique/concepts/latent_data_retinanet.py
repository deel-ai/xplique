import numpy as np
import tensorflow as tf
from .latent_extractor import LatentData

class TfLatentDataRetinanet(LatentData):
    index_activations = -1
    resnet_features: dict
    # typically a dict of resnet features with keys like 'P3', 'P4', 'P5'
    # Feature P3: np.shape (1, 80, 80, 512)
    # Feature P4: shape (1, 40, 40, 1024)
    # Feature P5: shape (1, 20, 20, 2048)
    image_shape: tuple

    def __init__(self, resnet_features, image_shape):
        self.resnet_features = resnet_features
        self.image_shape = image_shape

    def __len__(self) -> int:
        key = list(self.resnet_features.keys())[self.index_activations]
        return self.resnet_features[key].shape[0]

    def print_infos(self):
        print("LatentDataRetinanet:")
        for key, feature in self.resnet_features.items():
            print(f"\tFeature {key}: shape {feature.shape}")
        print(f"Image shape: {self.image_shape}")

    def get_activations(self):
        key = list(self.resnet_features.keys())[self.index_activations]
        activations = self.resnet_features[key]
        return activations

    def check_features_positive(self):
        for feature in self.resnet_features:
            if tf.reduce_any(feature < 0):
                raise ValueError("Features contain negative values, which is unexpected.")

    def set_activations(self, values):
        key = list(self.resnet_features.keys())[self.index_activations]
        if isinstance(values, tf.Tensor):
            self.resnet_features[key] = values
        elif isinstance(values, np.ndarray):
            self.resnet_features[key] = tf.convert_to_tensor(values)
        else:
            raise TypeError(f"Unsupported type: {type(values)}. Expected tf.Tensor or np.ndarray")

    def aggregate(self, *latent_data_list: 'LatentData') -> 'LatentData':
        print("[latent_data_retinanet] Aggregating LatentDataRetinanet... with", len(latent_data_list), "additional instances.")
        if not latent_data_list:
            return self
        # Aggregate the resnet dictionnaries (self.resnet_features + each latent_data_list.resnet_features))
        aggregated_resnet_features = {}
        for key in self.resnet_features.keys():
            all_features = [self.resnet_features[key]] + [data.resnet_features[key] for data in latent_data_list]
            aggregated_resnet_features[key] = tf.concat(all_features, axis=0)
        return TfLatentDataRetinanet(aggregated_resnet_features, self.image_shape)

    # used during the computing of the importances
    def __getitem__(self, index):
        """Allow indexing like latent_data[0] or slicing like latent_data[0:2]"""
        sliced_features = {}
        
        for key, feature in self.resnet_features.items():
            if isinstance(index, int):
                # Single index: latent_data[0]
                sliced_features[key] = feature[index:index+1]  # Garde dimension batch
            elif isinstance(index, slice):
                # Slice: latent_data[0:2] ou latent_data[:]
                sliced_features[key] = feature[index]
            else:
                raise TypeError(f"Index must be int or slice, got {type(index)}")
        
        return TfLatentDataRetinanet(sliced_features, self.image_shape)
        



        
    # def aggregate(self, *latent_data_list: 'LatentData') -> 'LatentData':
    #     # Aggregate the FPN features by concatenating them along the batch dimension
    #     aggregated_backbone_outputs = []
    #     for i, feature in enumerate(self.backbone_outputs):
    #         all_features = [feature] + [data.backbone_outputs[i] for data in latent_data_list]
    #         aggregated_backbone_outputs.append(tf.concat(all_features, axis=0))

    #     # Assuming all latent_data_list have the same image_shape, otherwise throw an error
    #     if not all(data.image_shape == latent_data_list[0].image_shape for data in latent_data_list):
    #         raise ValueError("All latent_data_list must have the same image_shape")
    #     # Check that the image_shape is consistent across all instances
    #     if not all(data.image_shape == self.image_shape for data in latent_data_list):
    #         raise ValueError("All latent_data_list must have the same image_shape")
    #     aggregated_image_shape = self.image_shape
    #     return TfLatentDataFpn(aggregated_backbone_outputs, aggregated_image_shape)



import types

from keras_cv.src.backend import ops
from keras_cv.src import bounding_box
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from xplique.concepts.latent_extractor import TfLatentExtractor
from xplique.utils_functions.object_detection.tf.box_formatter import RetinaNetProcessedBoxFormatter
# from xplique.utils_functions.box_manager import BoxFormat, BoxType
from typing import Callable

def buildTfRetinaNetLatentExtractor(model: Callable, nb_classes) -> 'TfLatentExtractor':

    def g(self, samples) -> TfLatentDataRetinanet:
        backbone_outputs = self.feature_extractor(samples, training=False)
        return TfLatentDataRetinanet(backbone_outputs, tuple(samples[0].shape))

    def h(self, latent_data: TfLatentDataRetinanet):
        backbone_outputs, image_shape = latent_data.resnet_features, latent_data.image_shape
        features = self.feature_pyramid(backbone_outputs, training=False)
        cls_outputs = []
        box_outputs = []
        for feature in features:
            cls_outputs.append(self.classification_head(feature))
            box_outputs.append(self.box_head(feature))
        
        # 4. Concatener les sorties
        cls_outputs = tf.concat([tf.reshape(c, [tf.shape(c)[0], -1, self.num_classes]) for c in cls_outputs], axis=1)
        box_outputs = tf.concat([tf.reshape(b, [tf.shape(b)[0], -1, 4]) for b in box_outputs], axis=1)
        
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

            #### fonction a remplacer: model.prediction_decoder()
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

    # model.decode_predictions_reworked = types.MethodType(decode_predictions_reworked, model)
    model.h = types.MethodType(h, model)
    model.g = types.MethodType(g, model)

    processed_formatter = RetinaNetProcessedBoxFormatter(nb_classes=nb_classes, input_image_size=(640, 640), output_image_size=(640, 640))
    latent_extractor = TfLatentExtractor(model, model.g, model.h, latent_data_class=TfLatentDataRetinanet, output_formatter=processed_formatter, batch_size=1)
    return latent_extractor