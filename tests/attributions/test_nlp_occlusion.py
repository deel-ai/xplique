"""
Test object detection BoundingBoxesExplainer
"""
import numpy as np

from xplique.attributions import NlpOcclusion

def test_masks():
    """Test the masks creation"""
    sentence = "aaa bbb ccc"
    words = sentence.split(" ")
    masks = NlpOcclusion._get_masks(words)
    assert masks.shape == (len(words), len(words))
    expected_mask = np.array([[False, True, True],
                              [True, False, True],
                              [True, True, False]])

    assert np.array_equal(masks, expected_mask)

def test_apply_masks():
    """Test if the application of a mask generate valid results"""
    sentence = "aaa bbb ccc"
    words = sentence.split(" ")
    masks = NlpOcclusion._get_masks(words)

    occluded_inputs = NlpOcclusion._apply_masks(words, masks)
    expected_occludec_inputs = [['bbb', 'ccc'], ['aaa', 'ccc'], ['aaa', 'bbb']]
    assert np.array_equal(occluded_inputs, expected_occludec_inputs)

def test_output_shape():
    """Test the output shape for several input sentences"""

    nb_concepts = 10

    def transform(inputs):
        # simulate the transorm method used in Craft/Cockatiel
        return np.ones((len(inputs), nb_concepts))

    input_sentence = ["aaa bbb ccc ddd eee fff", "ggg hhh iii jjj"]
    for sentence in input_sentence:
        words = sentence.split(" ")
        separator = " "

        method = NlpOcclusion(model=transform)
        sensitivity = method.explain(sentence, words, separator)

        assert sensitivity.shape == (nb_concepts, len(words))
