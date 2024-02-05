"""
COCKATIEL MANAGER Module for Pytorch
"""

from typing import Callable, List, Type
import numpy as np
import torch

from xplique.attributions.base import BlackBoxExplainer

from .craft_torch import BaseCraftManagerTorch
from .cockatiel import CockatielTorch
from ..commons import TokenExtractor, NlpPreprocessor
from ..commons.torch_operations import nlp_batch_predict


class BaseCockatielManagerTorch(BaseCraftManagerTorch):
    """
    Base class implementing the CockatielManager on Pytorch.
    This manager creates one Cockatiel instance per class to explain.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must return positive activations.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    preprocessor
        A callable object to transform strings into inputs for the model.
    inputs
        Input data: a list of strings.
    labels
        Labels of the inputs.
    list_of_class_of_interest
        A list of the classes id to explain. The manager will instanciate one
        CraftTorch object per element of this list.
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    device
        The type of device on which to place the torch tensors
    """

    def __init__(self, input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 preprocessor: NlpPreprocessor,
                 inputs: List[str],
                 labels: List[str],
                 list_of_class_of_interest: list = None,
                 number_of_concepts: int = 20,
                 batch_size: int = 64,
                 device: str = 'cuda'):

        super().__init__(input_to_latent_model=input_to_latent_model,
                         latent_to_logit_model=latent_to_logit_model,
                         inputs=inputs,
                         labels=labels,
                         list_of_class_of_interest=list_of_class_of_interest)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.device = device
        for class_of_interest in self.list_of_class_of_interest:
            craft = CockatielTorch(input_to_latent_model=input_to_latent_model,
                                   latent_to_logit_model=latent_to_logit_model,
                                   preprocessor=preprocessor,
                                   number_of_concepts=number_of_concepts,
                                   batch_size=batch_size,
                                   device=device)
            self.craft_instances[class_of_interest] = craft

    def compute_predictions(self) -> np.ndarray:
        """
        Computes predictions using the input-to-latent and latent-to-logit models,
        on the input data provided at the creation of the CockatielManager instance.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        features, _ = nlp_batch_predict(
            self.input_to_latent_model, preprocessor=self.preprocessor,
            inputs=self.inputs, labels=self.labels, batch_size=self.batch_size)
        logits = self.latent_to_logit_model(features)
        y_preds = np.array(torch.argmax(logits, -1).cpu().detach())
        return y_preds

    def get_best_excerpts_per_concept(self,
                                      class_id: int,
                                      nb_excerpts: int = 10,
                                      nb_most_important_concepts: int = None) -> List[str]:
        """
        Return the best excerpts for each concept.

        Parameters
        ----------
        class_id
            The class to explain.
        nb_excerpts
            The number of excerpts (patches) to fetch per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.

        Returns
        -------
        best_excerpts_per_concept
            The list of the best excerpts per concept
        """
        return self.craft_instances[class_id]\
            .get_best_excerpts_per_concept(nb_excerpts, nb_most_important_concepts)


class CockatielManagerNlpVisualizationMixin:
    """
    Class containing text visualization methods for CockatielManager.
    """

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
        self.craft_instances[class_id]\
            .plot_concepts_importances(importances=None,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=verbose)

    def display_concepts_excerpts(self,
                                  class_id: int,
                                  nb_patchs: int = 10,
                                  nb_most_important_concepts: int = None) -> None:
        """
        Display the best excerpts for each concept.

        Parameters
        ----------
        class_id
            The class to explain.
        nb_patchs
            The number of patches to display per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.
        """
        self.craft_instances[class_id]\
            .display_concepts_excerpts(nb_patchs=nb_patchs,
                                       nb_most_important_concepts=nb_most_important_concepts)

    def plot_concept_attribution_maps(self,
                                      class_id: int,
                                      sentences: List[str],
                                      token_extractor: TokenExtractor,
                                      explainer_class: Type[BlackBoxExplainer],
                                      ignore_words: List[str] = None,
                                      importances: np.ndarray = None,
                                      nb_most_important_concepts: int = 5,
                                      title: str = "",
                                      filter_percentile: int = 80,
                                      display_width: int = 400) -> List[str]:
        """
        Display the concepts attribution maps for the sentences given in argument.

        Parameters
        ----------
        class_id
            The class id to explain.
        sentences
            The list of sentences for which attribution maps will be displayed.
        token_extractor
            The token extractor used to extract tokens from the sentences.
        explainer_class
            Explainer class used during the masking process. Typically a NlpOcclusion class.
        ignore_words
            Words to ignore during the occlusion process. These will not be part of the
            extracted concepts.
        importances
            The importances computed by the estimate_importance() method.
            If None is provided, then the global importances will be used, otherwise
            the local importances set in this parameter will be used.
        nb_most_important_concepts
            Number of most important concepts to consider. Default is 5.
        title
            The title to use when displaying the results.
        filter_percentile
            Percentile used to filter the concept heatmap when displaying the HTML
            sentences (only show concept if excess N-th percentile). Default is 80.
        display_width
            Width of the displayed HTML text. Default is 400.

        Returns
        -------
        html_lines
            List of HTML text lines.
        """
        self.craft_instances[class_id].plot_concept_attribution_maps(
            sentences=sentences,
            token_extractor=token_extractor,
            explainer_class=explainer_class,
            ignore_words=ignore_words,
            importances=importances,
            nb_most_important_concepts=nb_most_important_concepts,
            title=title,
            filter_percentile=filter_percentile,
            display_width=display_width,
            css_namespace=class_id)


class CockatielManagerTorch(BaseCockatielManagerTorch, CockatielManagerNlpVisualizationMixin):
    """
    Class implementing CockatielManager on Pytorch, adapted for text visualization.
    This manager creates one CockatielTorch instance per class to explain.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must return positive activations.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    preprocessor
        A callable object to transform strings into inputs for the model.
    inputs
        Input data: a list of strings.
    labels
        Labels of the inputs of shape (n_samples, class_id)
    list_of_class_of_interest
        A list of the classes id to explain. The manager will instanciate one
        CraftTorch object per element of this list.
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    device
        The type of device on which to place the torch tensors
    """
