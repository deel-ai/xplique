"""
Functions to prepare `targets` for segmentation attributions
"""

import cv2
import numpy as np
import tensorflow as tf

from ..types import Union, Tuple, List


def get_class_zone(predictions: Union[tf.Tensor, np.array], class_id: int) -> tf.Tensor:
    """
    Extract a mask for the class `c`.
    The mask correspond to the pixels where the maximum prediction correspond to the class `c`.
    Other classes channels are set to zero.

    Parameters
    ----------
    predictions
        Output of the model, it should be the output of a softmax function.
        We assume the shape (h, w, c).
    class_id
        Index of the channel of the class of interest.

    Returns
    -------
    class_zone_mask
        Mask of the zone corresponding to the class of interest.
        Only the corresponding channel is non-zero.
        The shape is the same as `predictions`, (h, w, c).
    """
    assert len(tf.shape(predictions)) == 3, "predictions should correspond to only one image"

    class_zone = tf.cast(tf.argmax(predictions, axis=-1) == class_id, tf.float32)
    class_zone_mask = tf.Variable(tf.zeros(predictions.shape))
    class_zone_mask = class_zone_mask[:, :, class_id].assign(class_zone)

    assert tf.reduce_sum(class_zone_mask) >= 1

    return class_zone_mask


def get_connected_zone(
    predictions: Union[tf.Tensor, np.array], coordinates: Tuple[int, int]
) -> tf.Tensor:
    """
    Extract a connected mask around `coordinates`.
    The mask correspond to the pixels where the maximum prediction correspond
    to the maximum predicted class at `coordinates`.
    This class mask is then limited to the connected zone around `coordinates`.
    Other classes channels are set to zero.

    Parameters
    ----------
    predictions
        Output of the model, it should be the output of a softmax function.
        We assume the shape (h, w, c).
    coordinates
        Tuple of coordinates of the point inside the zone of interest.

    Returns
    -------
    connected_zone_mask
        Mask of the connected zone around `coordinates` with similar class prediction.
        Only the corresponding channel is non-zero.
        The shape is the same as `predictions`, (h, w, c).
    """
    assert len(tf.shape(predictions)) == 3, "predictions should correspond to only one image"

    assert (
        coordinates[0] < predictions.shape[0]
    ), f"Coordinates should be included in the shape, i.e. {coordinates[0]}<{predictions.shape[0]}"
    assert (
        coordinates[1] < predictions.shape[1]
    ), f"Coordinates should be included in the shape, i.e. {coordinates[1]}<{predictions.shape[1]}"

    labels = tf.argmax(predictions, axis=-1)
    class_id = labels[coordinates[0], coordinates[1]]
    mask = labels == class_id
    mask = np.uint8(np.array(mask)[:, :, np.newaxis] * 255)

    components_masks = cv2.connectedComponents(mask)[1]  # pylint: disable=no-member

    component_id = components_masks[coordinates[0], coordinates[1]]
    connected_zone = tf.cast(components_masks == component_id, tf.float32)

    connected_zone_mask = tf.Variable(tf.zeros(predictions.shape))
    connected_zone_mask = connected_zone_mask[:, :, class_id].assign(connected_zone)

    assert tf.reduce_sum(connected_zone_mask) >= 1
    assert connected_zone_mask[coordinates[0], coordinates[1], class_id] != 0

    return connected_zone_mask


def list_class_connected_zones(
        predictions: Union[tf.Tensor,np.array],
        class_id: int,
        zone_minimum_size: int = 100
) -> List[tf.Tensor]:
    """
    List all connected zones for a given class.
    A connected zone is a set of pixels next to each others
    where the maximum prediction correspond to the same class.
    This function generate a list of connected zones,
    each element of the list have a similar format to `get_connected_zone` outputs.

    Parameters
    ----------
    predictions
        Output of the model, it should be the output of a softmax function.
        We assume the shape (h, w, c).
    class_id
        Index of the channel of the class of interest.
    zone_minimum_size
        Threshold of number of pixels under which zones are not returned.

    Returns
    -------
    connected_zones_masks_list
        List of the connected zones masks for a given class.
        Each zone predictions shape is the same as `predictions`, (h, w, c).
        Only the corresponding channel is non-zero.
    """
    assert len(tf.shape(predictions)) == 3, "predictions should correspond to only one image"

    labels = tf.argmax(predictions, axis=-1)
    mask = labels == class_id
    mask = np.uint8(np.array(mask)[:, : , np.newaxis] * 255)

    components_masks = cv2.connectedComponents(mask)[1]  # pylint: disable=no-member

    sizes = np.bincount(components_masks.ravel())

    connected_zones_masks_list = []
    for component_id, size in enumerate(sizes[1:]):

        if size > zone_minimum_size:

            connected_zone = tf.cast(components_masks == (component_id + 1), tf.float32)

            all_channels_class_zone_mask = tf.Variable(tf.zeros(predictions.shape))
            all_channels_class_zone_mask =\
                all_channels_class_zone_mask[:, :, class_id].assign(connected_zone)

            assert tf.reduce_sum(all_channels_class_zone_mask) >= 1

            connected_zones_masks_list.append(all_channels_class_zone_mask)

    return connected_zones_masks_list



def get_in_out_border(
    class_zone_mask: Union[tf.Tensor, np.array],
) -> tf.Tensor:
    """
    Extract the border of a zone of interest, then put `1` on the
    inside border and `-1` on the outside border.

    Examples of coefficients extracted from the class channel of the class of interest:

    ```
    # class_zone_mask[:, :, c]
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 1, 1, 1]]

    # border_mask
    [[ 0,  0, -1, -1, -1],
     [ 0, -1, -1,  1,  1],
     [ 0, -1,  1,  1,  0],
     [-1, -1,  1,  0,  0],
     [-1,  1,  1,  0,  0]]
    ```

    Parameters
    ----------
    class_zone_mask
        Mask delimiting the zone of interest,  for the class of interest
        only one channel should have non-zero values,
        the one corresponding to the class.
        We assume the shape (h, w, c) same as the model output for one element.

    Returns
    -------
    class_borders_masks
        Mask of the borders of the zone of the class of interest.
        Only the corresponding channel is non-zero.
        Inside borders are set to `1` and outside borders are set to `-1`.
        The shape is the same as `class_zone_mask`, (h, w, c).
    """
    assert len(tf.shape(class_zone_mask)) == 3,\
        "class_zone_mask should correspond to only one image"

    # channel of the class of interest
    channel_mean = tf.reduce_sum(tf.cast(class_zone_mask, tf.int32), axis=[0, 1])
    assert (
        int(tf.reduce_sum(channel_mean)) >= 1
    ), "The specified `class_target_mask` is empty."
    class_id = int(tf.argmax(channel_mean))

    # set other values to -1 on the target zone to 1
    binary_mask = 2 * tf.cast(class_zone_mask[:, :, class_id], tf.int32) - 1

    # extend size with padding for convolution
    extended_binary_mask = tf.pad(
        binary_mask,
        tf.constant([[1, 1], [1, 1]]),
        "SYMMETRIC",
    )

    kernel = tf.convert_to_tensor(
        [[-1, -1, -1], [-1, 13, -1], [-1, -1, -1]], dtype=tf.int32
    )
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    conv_result = tf.nn.conv2d(
        tf.expand_dims(tf.expand_dims(extended_binary_mask, axis=0), axis=-1),
        kernel,
        strides=1,
        padding="VALID",
    )[0, :, :, 0]

    # 6 < in < 21, -21 < out < -6
    in_border = tf.logical_and(
        tf.math.less(tf.constant([6]), conv_result),
        tf.math.less(conv_result, tf.constant([21])),
    )
    out_border = tf.logical_and(
        tf.math.less(tf.constant([-21]), conv_result),
        tf.math.less(conv_result, tf.constant([-6])),
    )

    border_mask = (
        tf.zeros(binary_mask.shape)
        + tf.cast(in_border, tf.float32)
        - tf.cast(out_border, tf.float32)
    )

    class_borders_masks = tf.Variable(tf.zeros(class_zone_mask.shape))
    class_borders_masks = class_borders_masks[:, :, class_id].assign(border_mask)

    assert int(tf.reduce_sum(tf.abs(class_borders_masks))) >= 1

    return class_borders_masks


def get_common_border(
    border_mask_1: Union[tf.Tensor, np.array], border_mask_2: Union[tf.Tensor, np.array]
) -> tf.Tensor:
    """
    Compute the common part between `border_mask_1` and `border_mask_2` masks.
    Those borders should be computed using `get_in_out_border`.

    Parameters
    ----------
    border_mask_1
        Border of the first zone of interest. Computed with `get_in_out_border`.
    border_mask_2
        Border of the second zone of interest. Computed with `get_in_out_border`.

    Returns
    -------
    common_borders_masks
        Mask of the common borders between two zones of interest.
        Only the two corresponding channels are non-zero.
        Inside borders are set to `1` and outside borders are set to `-1`,
        Respectively on the two channels.
        The shape is the same as the input border masks, (h, w, c).
    """
    all_channel_border_mask_1 = tf.reduce_any(border_mask_1 != 0, axis=-1)
    all_channel_border_mask_2 = tf.reduce_any(border_mask_2 != 0, axis=-1)

    common_pixels_mask = tf.logical_and(
        all_channel_border_mask_1, all_channel_border_mask_2
    )

    assert tf.reduce_any(common_pixels_mask), "No common border between the two masks."

    return (border_mask_1 + border_mask_2) * tf.expand_dims(
        tf.cast(common_pixels_mask, tf.float32), -1
    )
