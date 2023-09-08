import tensorflow as tf

from xplique.utils_functions.segmentation import *

def get_prediction():
    predictions = [[[0.6, 0.6, 0.6, 0.2, 0.2],
                    [0.6, 0.6, 0.2, 0.2, 0.2],
                    [0.6, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],],
                   [[0.2, 0.2, 0.2, 0.6, 0.6],
                    [0.2, 0.2, 0.6, 0.6, 0.6],
                    [0.2, 0.6, 0.6, 0.6, 0.2],
                    [0.6, 0.6, 0.6, 0.2, 0.2],
                    [0.6, 0.6, 0.2, 0.2, 0.2],],
                   [[0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.6],
                    [0.2, 0.2, 0.2, 0.6, 0.6],
                    [0.2, 0.2, 0.6, 0.6, 0.6],],]
    
    predictions = tf.convert_to_tensor(predictions, tf.float32)

    predictions = tf.transpose(predictions, perm=[1, 2, 0])

    assert tf.reduce_all(tf.equal(tf.reduce_sum(predictions, axis=-1), tf.ones((5, 5))))

    return predictions


def test_get_class_zone():
    predictions = get_prediction()

    target_1 = get_class_zone(predictions, class_id=1)

    expected_target = tf.transpose(tf.convert_to_tensor(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],
         [[0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],],tf.float32), perm=[1, 2, 0])

    assert tf.reduce_all(tf.equal(target_1, expected_target))
    
    target_0 = get_class_zone(predictions, class_id=0)
    target_2 = get_class_zone(predictions, class_id=2)

    assert tf.reduce_all(tf.equal(
        tf.reduce_sum(tf.stack([target_0, target_1, target_2], axis=0), axis=[0, 3]),
        tf.fill((5, 5), 1.0)))


def test_get_connected_zone():
    predictions = get_prediction()

    target_1 = get_connected_zone(predictions, coordinates=(2, 2))

    expected_target = tf.transpose(tf.convert_to_tensor(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],
         [[0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],],tf.float32), perm=[1, 2, 0])

    assert tf.reduce_all(tf.equal(target_1, expected_target))
    
    target_0 = get_connected_zone(predictions, coordinates=(0, 0))
    target_2 = get_connected_zone(predictions, coordinates=(4, 4))

    assert tf.reduce_all(tf.equal(
        tf.reduce_sum(tf.stack([target_0, target_1, target_2], axis=0), axis=[0, 3]),
        tf.fill((5, 5), 1.0)))


def test_list_class_connected_zones():
    predictions = get_prediction()

    predictions = tf.stack([predictions[:, :, 0] + predictions[:, :, 2], predictions[:, :, 1]], axis=-1)

    zones_0 = list_class_connected_zones(predictions, class_id=0, zone_minimum_size=1)
    zones_1 = list_class_connected_zones(predictions, class_id=1, zone_minimum_size=1)
    no_zones = list_class_connected_zones(predictions, class_id=0, zone_minimum_size=10)

    assert len(zones_0) == 2
    assert len(zones_1) == 1
    assert len(no_zones) == 0

    expected_zones_1 = tf.transpose(tf.convert_to_tensor(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],
         [[0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0],],], tf.float32), perm=[1, 2, 0])
    
    assert tf.reduce_all(tf.equal(zones_1[0], expected_zones_1))

    expected_zones_21 = tf.transpose(tf.convert_to_tensor(
        [[[1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0],
          [1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],], tf.float32), perm=[1, 2, 0])

    expected_zones_22 = tf.transpose(tf.convert_to_tensor(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],], tf.float32), perm=[1, 2, 0])
    
    assert tf.reduce_all(tf.equal(zones_0[0], expected_zones_21))\
        or tf.reduce_all(tf.equal(zones_0[0], expected_zones_22))
    
    assert tf.reduce_all(tf.equal(zones_0[1], expected_zones_21))\
        or tf.reduce_all(tf.equal(zones_0[1], expected_zones_22))



def test_get_in_out_border():
    predictions = get_prediction()

    central_zone = get_connected_zone(predictions, coordinates=(2, 2))

    borders = get_in_out_border(central_zone)

    expected_borders = tf.transpose(tf.convert_to_tensor(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],
         [[ 0, -1, -1,1,  0],
          [-1, -1, 1, 1, 1],
          [-1, 1, 1,1 , -1],
          [ 1, 1, 1, -1, -1],
          [ 0,1, -1, -1,  0],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],],tf.float32), perm=[1, 2, 0])

    assert tf.reduce_all(tf.equal(borders, expected_borders))
    

def test_get_common_border():
    predictions = get_prediction()

    left_corner_zone = get_connected_zone(predictions, coordinates=(0, 0))
    central_zone = get_connected_zone(predictions, coordinates=(2, 2))

    left_corner_borders = get_in_out_border(left_corner_zone)
    central_borders = get_in_out_border(central_zone)

    common_borders_0 = get_common_border(left_corner_borders, central_borders)
    common_borders_1 = get_common_border(central_borders, left_corner_borders)

    expected_common_borders = tf.transpose(tf.convert_to_tensor(
        [[[ 0,  1,  1, -1,  0],
          [ 1,  1, -1, -1,  0],
          [ 1, -1, -1,  0,  0],
          [-1, -1,  0,  0,  0],
          [ 0,  0,  0,  0,  0],],
         [[ 0, -1, -1,  1,  0],
          [-1, -1,  1,  1,  0],
          [-1,  1,  1,  0,  0],
          [ 1,  1,  0,  0,  0],
          [ 0,  0,  0,  0,  0],],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],],],tf.float32), perm=[1, 2, 0])

    assert tf.reduce_all(tf.equal(common_borders_0, common_borders_1))

    assert tf.reduce_all(tf.equal(common_borders_0, expected_common_borders))