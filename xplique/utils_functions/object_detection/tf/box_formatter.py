from abc import ABC, abstractmethod
import tensorflow as tf

from xplique.utils_functions.object_detection.common.box_manager import BoxFormat, BoxType #, TfBoxManager
from xplique.utils_functions.object_detection.common.box_formatter import XpliqueBoxFormatter
from xplique.utils_functions.object_detection.tf.box_manager import TfBoxCoordinatesTranslator

class TfXpliqueBoxFormatter(XpliqueBoxFormatter, ABC):
    
    def __init__(self,
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True)):
        super().__init__(input_box_type=input_box_type, output_box_type=output_box_type)
        self.box_translator = TfBoxCoordinatesTranslator(self.input_box_type, self.output_box_type)
                                                   
    @abstractmethod
    def forward(self, predictions):
        raise NotImplementedError("This method should be implemented in the subclass")

    def format_predictions(self, predictions, input_image_size=None, output_image_size=None):
        boxes = predictions['boxes']
        boxes = self.box_translator.translate(boxes, input_image_size, output_image_size) # TODO add this also in torch
        probas = predictions['probas']
        scores = predictions['scores']#[..., tf.newaxis]
        return tf.concat([boxes, # boxes coordinates
                          scores, # detection probability
                          probas], # class logits predictions for the given box
                         axis=1)

class RetinaNetProcessedBoxFormatter(TfXpliqueBoxFormatter):

    def __init__(self, nb_classes: int,
                 input_box_type: BoxType = BoxType(BoxFormat.XYWH, is_normalized=False),
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
                 input_image_size: tuple = None,
                 output_image_size: tuple = None,
                 **kwargs):
        super().__init__(input_box_type, output_box_type)
        
        self.nb_classes = nb_classes
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size

    # eager mode forward method
    def forward_eager(self, predictions):
        results = []
        for boxes, scores, classes in zip(predictions['boxes'], predictions['confidence'], predictions['classes']):
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            # boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            scores = scores[:, tf.newaxis]
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict, self.input_image_size, self.output_image_size)
            results.append(formatted)
        result = tf.stack(results, axis=0)
        return result

    # graph mode forward pass
    def forward(self, predictions):
        def process_single_batch(batch_idx):
            boxes = predictions['boxes'][batch_idx]
            scores = predictions['confidence'][batch_idx] 
            classes = predictions['classes'][batch_idx]
            
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            scores_expanded = scores[:, tf.newaxis]
            
            pred_dict = {
                'boxes': boxes,
                'scores': scores_expanded,
                'probas': labels_one_hot
            }
            return self.format_predictions(pred_dict, self.input_image_size, self.output_image_size)

        batch_size = tf.shape(predictions['boxes'])[0]
        batch_indices = tf.range(batch_size)
        
        results = tf.map_fn(
            process_single_batch,
            batch_indices,
            fn_output_signature=tf.TensorSpec(shape=(None, 4 + 1 + self.nb_classes), dtype=tf.float32),
            parallel_iterations=1
        )
        
        return results

class TfSsdBoxFormatter(TfXpliqueBoxFormatter):

    def __init__(self, nb_classes: int,
                 input_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True)):
        super().__init__(input_box_type, output_box_type)
        self.nb_classes = nb_classes

    def forward_no_gradient_debug(self, predictions):
        results = []
        for boxes, scores, classes in zip(predictions['detection_boxes'], predictions['detection_scores'], predictions['detection_classes']):
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict)
            results.append(formatted)
        result = tf.stack(results, axis=0)
        return result

    def forward_no_gradient(self, predictions):
        def process_single_prediction(args):
            boxes, scores, classes = args
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['detection_boxes'],
            predictions['detection_scores'],
            predictions['detection_classes']
        )

        # Use tf.map_fn to process each batch element
        results = tf.map_fn(
            process_single_prediction,
            stacked_inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            parallel_iterations=10
        )
        return results

    def forward(self, predictions):
        def process_single_prediction(args):
            boxes, scores = args
            probas = tf.nn.softmax(scores, axis=-1)
            detection_scores = tf.reduce_max(probas, axis=-1, keepdims=True)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': detection_scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['raw_detection_boxes'],    # (1, 1917, 4)
            predictions['raw_detection_scores']    # (1, 1917, 91)
        )

        # Use tf.map_fn to process each batch element
        results = tf.map_fn(
            process_single_prediction,
            stacked_inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            parallel_iterations=10
        )
        return results

    # Forward with raw results
    def forward_debug(self, predictions):
        def process_single_prediction(args):
            boxes, scores = args
            probas = tf.nn.softmax(scores, axis=-1)
            detection_scores = tf.reduce_max(probas, axis=-1, keepdims=True)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': detection_scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['raw_detection_boxes'],    # (1, 1917, 4)
            predictions['raw_detection_scores']    # (1, 1917, 91)
        )

        results = []
        for args in zip(*stacked_inputs):
            formatted = process_single_prediction(args)
            results.append(formatted)
        results = tf.stack(results, axis=0)
        return results
