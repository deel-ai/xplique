import tensorflow as tf

from xplique.utils_functions.object_detection.common.box_manager import BoxManager, BoxFormat, BoxType

class TfBoxManager(BoxManager):

    def __init__(self, format: BoxFormat, normalized: bool):
        self.format = format
        self.normalized = normalized

    @staticmethod
    def normalize_boxes(raw_boxes: tf.Tensor, image_source_size: tf.Tensor):
        sx, sy = image_source_size
        normalized_boxes = tf.identity(raw_boxes)
        normalized_boxes = tf.concat(
            [
                normalized_boxes[:, 0:1] / sx,
                normalized_boxes[:, 1:2] / sy,
                normalized_boxes[:, 2:3] / sx,
                normalized_boxes[:, 3:4] / sy,
            ],
            axis=1,
        )
        # logging.debug("Normalized boxes: %s", normalized_boxes[0:1])
        return normalized_boxes

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: tf.Tensor):
        x_c, y_c, w, h = tf.unstack(normalized_boxes, axis=1)
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        # x_min = x_c - w
        # y_min = y_c - h
        # x_max = x_c + w
        # y_max = y_c + h
        b = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        # logging.debug("Converted from CXCYWH to XYXY: %s", b[0])
        return b
    
    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: tf.Tensor):
        x, y, w, h = tf.unstack(normalized_boxes, axis=1) # extract the columns
        b = [x, y, x + w, y + h]
        # logging.debug("Converted from XYWH to XYXY: %s", b[0][0])
        return tf.stack(b, axis=1)
    
    @staticmethod
    def rescale_bboxes(boxes: tf.Tensor, size: tf.Tensor):
        img_w, img_h = size
        rescaled_boxes = boxes * tf.constant(
            [img_w, img_h, img_w, img_h], dtype=tf.float32
        )
        # logging.debug("Rescaled boxes: %s", rescaled_boxes[0])
        return rescaled_boxes

    # Resize boxes in the input source size dimension, to output targe size dimension.
    # The input box format can be either CXCYWH or XYXY, the output format is XYXY (always), for Craft compatibility.

    def resize(
        self, raw_boxes: tf.Tensor, image_source_size: tuple, image_target_size: tuple
    ):
        # if self.normalized:
        #     normalized_boxes = TfBoxManager.normalize_boxes(
        #         raw_boxes, image_source_size
        #     )
        # else:
        #     normalized_boxes = raw_boxes

        # @TODO: normalization issue
        normalized_boxes = raw_boxes

        if self.format == BoxFormat.CXCYWH:
            xyxy_boxes = TfBoxManager.box_cxcywh_to_xyxy(normalized_boxes)
        elif self.format == BoxFormat.XYWH:
            xyxy_boxes = TfBoxManager.box_xywh_to_xyxy(normalized_boxes)
        else:
            xyxy_boxes = normalized_boxes

        # scaled_boxes = TfBoxManager.rescale_bboxes(xyxy_boxes, image_target_size)
        scaled_boxes = xyxy_boxes  # @TODO normalization issue
        return scaled_boxes



class TfBoxCoordinatesTranslator:
    def __init__(self, input_box_type: BoxType, output_box_type: BoxType):
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type

    def translate(self, box: tf.Tensor, input_image_size: tf.TensorShape = None, output_image_size: tf.TensorShape = None):
        # logging.info("Translating box from %s (size: %s) to %s (size: %s)", self.input_box_type, input_image_size, self.output_box_type, output_image_size)
        # logging.debug("Input box: %s", box[0])
        if not isinstance(box, tf.Tensor):
            box = tf.convert_to_tensor(box)

        # normalize the input box if needed
        if not self.input_box_type.is_normalized:
            if input_image_size is None:
                raise ValueError("Input image size must be provided for non-normalized boxes.")
            box = TfBoxManager.normalize_boxes(box, input_image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TfBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = TfBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TfBoxManager.box_xyxy_to_cxcywh(box) # TODO: add this method
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            box = TfBoxManager.box_xyxy_to_xywh(box) # TODO: add this method

        # rescale the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            if output_image_size is None:
                raise ValueError("Output image size must be provided for non-normalized boxes.")
            box = TfBoxManager.rescale_bboxes(box, output_image_size)

        return box

def filter_boxes_same_dim(model_output_nbc: list, class_id: int, accuracy: float=0.55) -> list:
    filtered = []
    for nbc_annotations in model_output_nbc:
        probas = nbc_annotations[:, 5:]
        class_ids = tf.argmax(probas, axis=1)
        scores = nbc_annotations[:, 4]
        keep = (class_ids == class_id) & (scores > accuracy)
        # replace nbc_annotations where keep is False with zeros
        zeros = tf.zeros_like(nbc_annotations)
        nbc_annotations_filtered = tf.where(keep[:, tf.newaxis], nbc_annotations, zeros)
        
        filtered.append(nbc_annotations_filtered)
    return filtered

def filter_boxes(model_output_nbc: list, class_id: int = None, accuracy: float = None) -> list:

    if class_id is None and accuracy is None:
        return model_output_nbc
    filtered = []
    for nbc_annotations in model_output_nbc:
        probas = nbc_annotations[:, 5:]
        class_ids = tf.argmax(probas, axis=1)
        scores = nbc_annotations[:, 4]
        if class_id is None:
            keep = scores > accuracy
        elif accuracy is None:
            keep = class_ids == class_id
        else:
            keep = (class_ids == class_id) & (scores > accuracy)
        filtered.append(nbc_annotations[keep])
    # filtered contains a list of nbc_matrix of different sizes
    return filtered
