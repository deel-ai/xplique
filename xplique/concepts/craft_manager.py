"""
CRAFT MANAGER Module for Tensorflow/Pytorch
"""

from abc import ABC, abstractmethod

import numpy as np

from ..types import Callable, Optional
from .craft import DisplayImportancesOrder

class BaseCraftManager(ABC):
    """
    Base class implementing the CRAFT Concept Extraction Mechanism on multiple classes.
    This manager creates one BaseCraft instance per class to explain.

    It shall be subclassed in order to adapt to a specific framework
    (Tensorflow or Pytorch)

    Parameters
    ----------
    inputs
        Input data of shape (n_samples, height, width, channels).
        (x1, x2, ..., xn) in the paper.
    labels
        Labels of the inputs of shape (n_samples, class_id)
    list_of_class_of_interest
        A list of the classes id to explain. The manager will instanciate one
        Craft object per element of this list.
    """

    @abstractmethod
    def __init__(self,
                input_to_latent_model : Callable,
                latent_to_logit_model : Callable,
                inputs : np.ndarray,
                labels : np.ndarray,
                list_of_class_of_interest : Optional[list] = None):
        self.input_to_latent_model = input_to_latent_model
        self.latent_to_logit_model = latent_to_logit_model
        self.inputs = inputs
        self.labels = labels

        if list_of_class_of_interest is None:
            # take all the classes
            list_of_class_of_interest = np.array(list(set(labels)))
        self.list_of_class_of_interest = list_of_class_of_interest
        self.craft_instances = None

    @abstractmethod
    def compute_predictions(self):
        """
        Compute the predictions for the current dataset, using the 2 models
        input_to_latent_model and latent_to_logit_model chained.
        To be implemented by decicated subclasses for Tensorflow or Pytorch.

        Returns
        -------
        y_preds
            the predictions
        """
        raise NotImplementedError

    def fit(self, nb_samples_per_class: Optional[int] = None, verbose: bool = False):
        """
        Fit the Craft models on their respective class of interest.

        Parameters
        ----------
        nb_samples_per_class
            Number of samples to use to fit the Craft model.
            Default is None, which means that all the samples will be used.
        verbose
            If True, then print the current class CRAFT is fitting,
            otherwise no textual output will be printed.
        """
        y_preds = self.compute_predictions()

        for class_of_interest, craft_instance in self.craft_instances.items():
            if verbose:
                print(f'Fitting CRAFT instance for class {class_of_interest} ')
            filtered_indices = np.where(y_preds == class_of_interest)
            class_inputs = self.inputs[filtered_indices]
            class_labels = self.labels[filtered_indices]
            if nb_samples_per_class is not None:
                class_inputs = class_inputs[:nb_samples_per_class]
                class_labels = class_labels[:nb_samples_per_class]
            craft_instance.fit(class_inputs, class_id=class_of_interest)

    def estimate_importance(self, nb_design: int = 32, verbose: bool = False):
        """
        Estimates the importance of each concept for all the classes of interest.

        Parameters
        ----------
        nb_design
            The number of design to use for the importance estimation. Default is 32.
        verbose
            If True, then print the current class CRAFT is estimating importances for,
            otherwise no textual output will be printed.
        """
        for class_of_interest, craft_instance in self.craft_instances.items():
            if verbose:
                print(f'Estimating importances for class {class_of_interest} ')
            craft_instance.estimate_importance(nb_design=nb_design)

    def plot_concepts_importances(self,
                                  class_id: int,
                                  nb_most_important_concepts: int = 5,
                                  verbose: bool = False):
        """
        Plot a bar chart displaying the importance value of each concept.

        Parameters
        ----------
        class_id
            The class to explain.
        nb_most_important_concepts
            The number of concepts to focus on. Default is 5.
        verbose
            If True, then print the importance value of each concept, otherwise no textual
            output will be printed.
        """
        self.craft_instances[class_id].plot_concepts_importances(importances = None,
                                        nb_most_important_concepts=nb_most_important_concepts,
                                        verbose=verbose)

    def plot_concepts_crops(self,
                            class_id: int, nb_crops: int = 10,
                            nb_most_important_concepts: int = None):
        """
        Display the crops for each concept.

        Parameters
        ----------
        class_id
            The class to explain.
        nb_crops
            The number of crops (patches) to display per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.
        """
        self.craft_instances[class_id].plot_concepts_crops(nb_crops=nb_crops,
                        nb_most_important_concepts=nb_most_important_concepts)

    def plot_image_concepts(self,
                            img: np.ndarray,
                            class_id: int,
                            display_importance_order: DisplayImportancesOrder = \
                                                    DisplayImportancesOrder.GLOBAL,
                            nb_most_important_concepts: int = 5,
                            filter_percentile: int = 90,
                            clip_percentile: Optional[float] = 10,
                            alpha: float = 0.65,
                            filepath: Optional[str] = None):
        """
        All in one method displaying several plots for the image `id` given in argument:
        - the concepts attribution map for this image
        - the best crops for each concept (displayed around the heatmap)
        - the importance of each concept

        Parameters
        ----------
        img
            The image to explain.
        class_id
            The class to explain.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
            Default to GLOBAL.
        nb_most_important_concepts
            The number of concepts to focus on. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        filepath
            Path the file will be saved at. If None, the function will call plt.show().
        """
        self.craft_instances[class_id].plot_image_concepts(img,
                                            display_importance_order=display_importance_order,
                                            nb_most_important_concepts=nb_most_important_concepts,
                                            filter_percentile=filter_percentile,
                                            clip_percentile=clip_percentile,
                                            alpha=alpha,
                                            filepath=filepath)
