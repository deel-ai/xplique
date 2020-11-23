"""
Module related to the Concept extraction mechanism
"""

import tensorflow as tf
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


class Cav: # pylint: disable=too-few-public-methods
    """
    Used to compute the Concept Activation Vector, which is a vector in the direction of the
    activations of that concept’s set of examples.

    Ref. Kim & al., Interpretability Beyond Feature Attribution: Quantitative Testing with Concept
    Activation Vectors (TCAV) (2018).
    https://arxiv.org/abs/1711.11279

    Parameters
    ----------
    model : tf.keras.Model
        Model to extract concept from.
    target_layer : int or string
        Index of the target layer or name of the layer.
    classifier : 'SGD' or 'SVM' or Sklearn model, optional
        Default implementation use SGD classifier, SVM give more robust results but the computation
        time is longer.
    test_fraction : float, optional
        Fraction of the dataset used for test
    batch_size : int, optional
        Batch size during the activations extraction
    verbose : boolean, optional
        If true, display information while training the classifier
    """

    def __init__(self, model, target_layer, classifier='SGD', test_fraction=0.2, batch_size=64,
                 verbose=False):
        self.model = model
        self.batch_size = batch_size
        self.test_fraction = test_fraction
        self.verbose = verbose

        # configure model bottleneck
        target_layer = model.get_layer(target_layer).output if isinstance(target_layer, str) else \
            model.layers[target_layer].output
        self.bottleneck_model = tf.keras.Model(model.input, target_layer)

        # configure classifier
        if classifier == 'SGD':
            self.classifier = SGDClassifier(loss='hinge',
                                            penalty='l2',
                                            validation_fraction=0.2,
                                            verbose=self.verbose)
        elif classifier == 'SVM':
            self.classifier = LinearSVC(verbose=self.verbose)
        elif all(hasattr(classifier, attr) for attr in ['fit', 'score']):
            self.classifier = classifier
        else:
            raise ValueError('The classifier passed is invalid.')

    def fit(self, positive_dataset, negative_dataset):
        """
        Compute and return the Concept Activation Vector (CAV) associated to the dataset and the
        layer targeted.

        Parameters
        ----------
        positive_dataset : ndarray (N, W, H, C)
            Dataset of positive samples : samples containing the concept.
        negative_dataset : ndarray (N, W, H, C)
            Dataset of negative samples : samples without the concept

        Returns
        -------
        cav : ndarray
            Vector of the same shape as the layer output
        """
        positive_activations = self.bottleneck_model.predict(positive_dataset,
                                                             batch_size=self.batch_size)
        negative_activations = self.bottleneck_model.predict(negative_dataset,
                                                             batch_size=self.batch_size)

        activations = np.concatenate([positive_activations, negative_activations], axis=0)
        has_concept = np.array(
            [1 for _ in positive_activations] + [0 for _ in negative_activations], np.float32)

        original_shape = activations.shape
        activations = activations.reshape(len(activations), -1)

        x_train, x_test, y_train, y_test = train_test_split(activations, has_concept,
                                                            test_size=self.test_fraction)

        self.classifier.fit(x_train, y_train)

        val_accuracy = self.classifier.score(x_test, y_test)
        if self.verbose:
            print(f"[CAV] val_accuracy : {round(float(val_accuracy * 100), 2)}")

        # weights of each feature is the vector orthogonal to the hyperplane
        # note : x + epsilon * cav should increase the concept
        cav = self.classifier.coef_[0]
        cav = cav / np.linalg.norm(cav, 1)
        cav = np.array(cav).reshape(original_shape[1:])

        return cav

    __call__ = fit
