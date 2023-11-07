"""
CRAFT Module for Tensorflow/Pytorch
"""

from abc import ABC, abstractmethod
import dataclasses
from enum import Enum
import colorsys
from math import ceil

import numpy as np
import cv2
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import gridspec

from xplique.attributions.global_sensitivity_analysis import (HaltonSequenceRS, JansenEstimator)
from xplique.plots.image import _clip_percentile

from ..types import Callable, Tuple, Optional, Union
from .base import BaseConceptExtractor

@dataclasses.dataclass
class Factorization:
    """ Dataclass handling data produced during the Factorization step."""
    inputs: np.ndarray
    class_id: int
    crops: np.ndarray
    reducer: NMF
    crops_u: np.ndarray
    concept_bank_w: np.ndarray

class Sensitivity:
    """
    Dataclass handling data produced during the Sobol indices computation.
    This is an internal data class used by Craft to store computation data.

    Parameters
    ----------
    importances
        The Sobol total index (importance score) for each concept.
    most_important_concepts
        The number of concepts to display. If None is provided, then all the concepts
        will be displayed unordered, otherwise only nb_most_important_concepts will be
        displayed, ordered by importance.
    cmaps
        The list of colors associated with each concept.
        Can be either:
            - A list of (r, g, b) colors to use as a base for the colormap.
            - A colormap name compatible with `plt.get_cmap(cmap)`.
    """

    def __init__(self, importances: np.ndarray,
                       most_important_concepts: np.ndarray,
                       cmaps: Optional[Union[list, str]]=None):
        self.importances = importances
        self.most_important_concepts = most_important_concepts
        self.set_concept_attribution_cmap(cmaps=cmaps)

    @staticmethod
    def _get_alpha_cmap(cmap: Union[Tuple,str]):
        """
        Creat a colormap with an alpha channel, out of 3 (r, g, b) values.
        This is used in particular by `set_concept_attribution_cmap()` to
        create different colors for each concept.

        Parameters
        ----------
        cmap
            The color associated with each concept.
            Can be either:
                - A tuple (r, g, b) of color to use as a base for the colormap.
                - A colormap name compatible with `plt.get_cmap(cmap)`.
        Returns
        -------
        colormap
            An instance of ListedColormap with an alpha channel.
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        else:
            colors = np.array(cmap)
            if np.any(colors > 1.0):
                colors = colors / 255.0

            cmax = colorsys.rgb_to_hls(*colors)
            cmax = np.array(cmax)
            cmax[-1] = 1.0

            cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
            cmap = LinearSegmentedColormap.from_list("", [colors, cmax])

        alpha_cmap = cmap(np.arange(256))
        alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
        alpha_cmap = ListedColormap(alpha_cmap)

        return alpha_cmap

    def set_concept_attribution_cmap(self, cmaps: Optional[Union[Tuple, str]]=None):
        """
        Set the colormap used for the concepts displayed in the attribution maps.

        Parameters
        ----------
        cmaps
            The list of colors associated with each concept.
            Can be either:
                - A list of (r, g, b) colors to use as a base for the colormap.
                - A colormap name compatible with `plt.get_cmap(cmap)`.
        """
        if cmaps is None:
            self.cmaps = [
                    Sensitivity._get_alpha_cmap((54, 197, 240)),
                    Sensitivity._get_alpha_cmap((210, 40, 95)),
                    Sensitivity._get_alpha_cmap((236, 178, 46)),
                    Sensitivity._get_alpha_cmap((15, 157, 88)),
                    Sensitivity._get_alpha_cmap((84, 25, 85)),
                    Sensitivity._get_alpha_cmap((55, 35, 235))
                ]
            # Add more colors by default
            cmaps_more = [Sensitivity._get_alpha_cmap(cmap)
                            for cmap in plt.get_cmap('tab10').colors]
            self.cmaps.extend(cmaps_more)
        else:
            self.cmaps = [Sensitivity._get_alpha_cmap(cmap) for cmap in cmaps]

        if len(self.cmaps) < len(self.most_important_concepts):
            raise RuntimeError(f'Not enough colors in cmaps ({len(self.cmaps)}) ' \
                               f'compared to the number of important concepts ' \
                               '({len(self.most_important_concepts)})')

class DisplayImportancesOrder(Enum):
    """
    Select in which order the concepts will be displayed.

    GLOBAL: Order concepts by their global importance on the whole dataset
    LOCAL: Order concepts by their Local importance on a single sample
    """
    GLOBAL = 0
    LOCAL  = 1

    def __eq__(self, other):
        return self.value == other.value

class BaseCraft(BaseConceptExtractor, ABC):
    """
    Base class implementing the CRAFT Concept Extraction Mechanism.

    Ref. Fel et al.,  CRAFT Concept Recursive Activation FacTorization (2023).
    https://arxiv.org/abs/2211.10154

    It shall be subclassed in order to adapt to a specific framework
    (Tensorflow or Pytorch)

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must return positive activations.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches (crops) to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent_model : Callable,
                       latent_to_logit_model : Callable,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64):
        super().__init__(number_of_concepts, batch_size)
        self.input_to_latent_model = input_to_latent_model
        self.latent_to_logit_model = latent_to_logit_model
        self.patch_size = patch_size

        self.factorization = None
        self.sensitivity = None

        # sanity checks
        assert(hasattr(input_to_latent_model, "__call__")), \
               "input_to_latent_model must be a callable function"
        assert(hasattr(latent_to_logit_model, "__call__")), \
               "latent_to_logit_model must be a callable function"


    @abstractmethod
    def _latent_predict(self, inputs: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _logit_predict(self, activations: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _extract_patches(self, inputs: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _to_np_array(self, inputs: np.ndarray, dtype: type) -> np.ndarray:
        raise NotImplementedError

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """
        if self.factorization is None:
            raise NotFittedError("The factorization model has not been fitted to input data yet.")

    def fit(self,
            inputs : np.ndarray,
            class_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, height, width, channels).
            (x1, x2, ..., xn) in the paper.
        class_id
            The class id of the inputs.

        Returns
        -------
        crops
            The crops (X in the paper)
        crops_u
            The concepts' values (U in the paper)
        concept_bank_w
            The concept's basis (W in the paper)

        """
        # extract patches (crops) from the input data
        crops, activations = self._extract_patches(inputs)

        # apply NMF to the activations to obtain matrices U and W
        reducer = NMF(n_components=self.number_of_concepts, alpha_W=1e-2)
        crops_u = reducer.fit_transform(activations)
        concept_bank_w = reducer.components_.astype(np.float32)

        self.factorization = Factorization(inputs, class_id, crops,
                                           reducer, crops_u, concept_bank_w)

        return crops, crops_u, concept_bank_w

    def transform(self, inputs : np.ndarray, activations : np.ndarray = None) -> np.ndarray:
        """Transforms the inputs data into its concept representation.

        Parameters
        ----------
        inputs
            The input data to be transformed.
        activations
            Pre-computed activations of the input data. If not provided, the activations
            will be computed using the input_to_latent_model model on the inputs.

        Returns
        -------
        coeffs_u
            The concepts' values of the inputs (U in the paper).
        """
        self.check_if_fitted()

        if activations is None:
            activations = self._latent_predict(inputs)

        is_4d = len(activations.shape) == 4

        if is_4d:
            # (N, W, H, C) -> (N * W * H, C)
            original_shape = activations.shape[:-1]
            activations = np.reshape(activations, (-1, activations.shape[-1]))

        w_dtype = self.factorization.reducer.components_.dtype
        coeffs_u = self.factorization.reducer.transform(self._to_np_array(activations,
                                                                          dtype=w_dtype))

        if is_4d:
            # (N * W * H, R) -> (N, W, H, R) with R = nb_concepts
            coeffs_u = np.reshape(coeffs_u, (*original_shape, coeffs_u.shape[-1]))
        return coeffs_u

    def estimate_importance(self, inputs : np.ndarray = None, nb_design: int = 32) -> np.ndarray:
        """
        Estimates the importance of each concept for a given class, either globally
        on the whole dataset provided in the fit() method (in this case, inputs shall
        be set to None), or locally on a specific input image.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data on which to compute the importances.
            If None, then the inputs provided in the fit() method
            will be used (global importance of the whole dataset).
            Default is None.
        nb_design
            The number of design to use for the importance estimation. Default is 32.

        Returns
        -------
        importances
            The Sobol total index (importance score) for each concept.

        """
        self.check_if_fitted()

        compute_global_importances = False
        if inputs is None:
            inputs = self.factorization.inputs
            compute_global_importances = True

        coeffs_u = self.transform(inputs)

        masks = HaltonSequenceRS()(self.number_of_concepts, nb_design = nb_design)
        estimator = JansenEstimator()

        importances = []

        if len(coeffs_u.shape) == 2:
            # apply the original method of the paper

            for coeff in coeffs_u:
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.factorization.concept_bank_w

                y_pred = self._logit_predict(a_perturbated)
                y_pred = y_pred[:, self.factorization.class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        elif len(coeffs_u.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for coeff in coeffs_u:
                u_perturbated = coeff[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(u_perturbated,
                                        (-1, coeff.shape[-1])) @ self.factorization.concept_bank_w
                a_perturbated = np.reshape(a_perturbated,
                                        (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))

                # a_perturbated: (N, H, W, C)
                y_pred = self._logit_predict(a_perturbated)
                y_pred = y_pred[:, self.factorization.class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        importances = np.mean(importances, 0)

        # Save the results of the computation if working on the whole dataset
        if compute_global_importances:
            most_important_concepts = np.argsort(importances)[::-1]
            self.sensitivity = Sensitivity(importances, most_important_concepts)

        return importances

    def plot_concepts_importances(self,
                                  importances: np.ndarray = None,
                                  display_importance_order: DisplayImportancesOrder = \
                                                    DisplayImportancesOrder.GLOBAL,
                                  nb_most_important_concepts: int = None,
                                  verbose: bool = False):
        """
        Plot a bar chart displaying the importance value of each concept.

        Parameters
        ----------
        importances
            The importances computed by the estimate_importance() method.
            Default is None, in this case the importances computed on the whole
            dataset will be used.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
        nb_most_important_concepts
            The number of concepts to display. If None is provided, then all the concepts
            will be displayed unordered, otherwise only nb_most_important_concepts will be
            displayed, ordered by importance.
            Default is None.
        verbose
            If True, then print the importance value of each concept, otherwise no textual
            output will be printed.
        """

        if importances is None:
            # global
            importances = self.sensitivity.importances
            most_important_concepts = self.sensitivity.most_important_concepts
        else:
            # local
            most_important_concepts = np.argsort(importances)[::-1]

        if nb_most_important_concepts is None:
            # display all concepts not ordered
            global_color_index_order = self.sensitivity.most_important_concepts
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in range(len(importances))]
            colors = np.array([colors(1.0)
                               for colors in self.sensitivity.cmaps])[local_color_index_order]

            plt.bar(range(len(importances)), importances, color=colors)
            plt.xticks(range(len(importances)))

        else:
            # only display the nb_most_important_concepts concepts in their importance order
            most_important_concepts = most_important_concepts[:nb_most_important_concepts]

            # Find the correct color index
            global_color_index_order = np.argsort(self.sensitivity.importances)[::-1]
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in most_important_concepts]
            colors = np.array([colors(1.0)
                               for colors in self.sensitivity.cmaps])[local_color_index_order]

            plt.bar(range(len(importances[most_important_concepts])),
                    importances[most_important_concepts], color=colors)
            plt.xticks(ticks=range(len(most_important_concepts)),
                       labels=most_important_concepts)

        if display_importance_order == DisplayImportancesOrder.GLOBAL:
            importance_order = "Global"
        else:
            importance_order = "Local"
        plt.title(f"{importance_order} Concept Importance")

        if verbose:
            for c_id in most_important_concepts:
                print(f"Concept {c_id} has an importance value of {importances[c_id]:.2f}")

    @staticmethod
    def _show(img, **kwargs):
        img = np.array(img)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        plt.imshow(img, **kwargs)
        plt.axis('off')

    def plot_concepts_crops(self,
                            nb_crops: int = 10,
                            nb_most_important_concepts: int = None,
                            verbose: bool = False) -> None:
        """
        Display the crops for each concept.

        Parameters
        ----------
        nb_crops
            The number of crops (patches) to display per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.
        verbose
            If True, then print the importance value of each concept,
            otherwise no textual output will be printed.
        """
        most_important_concepts = self.sensitivity.most_important_concepts
        if nb_most_important_concepts is not None:
            most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        for c_id in most_important_concepts:
            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]

            if verbose:
                print(f"Concept {c_id} has an importance value of " \
                    f"{self.sensitivity.importances[c_id]:.2f}")
            plt.figure(figsize=(7, (2.5/2)*ceil(nb_crops/5)))
            for i in range(nb_crops):
                plt.subplot(ceil(nb_crops/5), 5, i+1)
                BaseCraft._show(best_crops[i])
            plt.show()

    def plot_concept_attribution_legend(self,
                                        nb_most_important_concepts: int = 6,
                                        border_width: int=5):
        """
        Plot a legend for the concepts attribution maps.

        Parameters
        ----------
        nb_most_important_concepts
            The number of concepts to focus on. Default is 6.
        border_width
            Width of the border around each concept image, in pixels. Defaults to 5.
        """

        most_important_concepts = \
            self.sensitivity.most_important_concepts[:nb_most_important_concepts]
        for i, c_id in enumerate(most_important_concepts):
            cmap = self.sensitivity.cmaps[i]
            plt.subplot(1, len(most_important_concepts), i+1)

            best_crops_id = np.argsort(self.factorization.crops_u[:, c_id])[::-1][0]
            best_crop = self.factorization.crops[best_crops_id]

            if best_crop.shape[0] > best_crop.shape[-1]:
                mask = np.zeros(best_crop.shape[:-1]) # tf
            else:
                mask = np.zeros(best_crop.shape[1:]) # torch
            mask[:border_width, :] = 1.0
            mask[:, :border_width] = 1.0
            mask[-border_width:, :] = 1.0
            mask[:, -border_width:] = 1.0

            BaseCraft._show(best_crop)
            BaseCraft._show(mask, cmap=cmap)
            plt.title(f"{c_id}", color=cmap(1.0))

        plt.show()

    @staticmethod
    def compute_subplots_layout_parameters(images: np.ndarray,
                                           cols: int = 5,
                                           img_size: float = 2.,
                                           margin: float = 0.3,
                                           spacing: float = 0.3):
        """
        Compute layout parameters for subplots, to be used by the
        method fig.subplots_adjust()

        Parameters
        ----------
        images
            The images to display with subplots. Should be
            data of shape (n_samples, height, width, channels).
        cols
            Number of columns to configure for the subplots.
            Defaults to 5.
        img_size
            Size of each subplots (in inch), considering we keep aspect ratio. Defaults to 2.
        margin
            The margin to use for the subplots. Defaults to 0.3.
        spacing
            The spacing to use for the subplots. Defaults to 0.3.

        Returns
        -------
        layout_parameters
            A dictionary containing the layout description
        rows
            The number of rows needed to display the images
        figwidth
            The figures width in the subplots
        figheight
            The figures height in the subplots

        """
        # get width and height of our images
        if images.shape[1] == 3:
            nb_samples = images.shape[0]
            [l_width, l_height] = images.shape[2:4] # pytorch
        else:
            [nb_samples, l_width, l_height] = images.shape[0:3] # tf
        rows = ceil(nb_samples / cols)

        # define the figure margin, width, height in inch
        figwidth = cols * img_size + (cols-1) * spacing + 2 * margin
        figheight = rows * img_size * l_height/l_width + (rows-1) * spacing + 2 * margin

        layout_parameters = {
            'left'      : margin/figwidth,
            'right'     : 1.-(margin/figwidth),
            'bottom'    : margin/figheight,
            'top'       :  1.-(margin/figheight),
            'wspace'    : spacing/img_size,
            'hspace'    : spacing/img_size * l_width/l_height
        }
        return layout_parameters, rows, figwidth, figheight

    def plot_concept_attribution_maps(self,
                                      images: np.ndarray,
                                      importances: np.ndarray = None,
                                      nb_most_important_concepts: int = 5,
                                      filter_percentile: int = 90,
                                      clip_percentile: Optional[float] = 10.0,
                                      alpha: float = 0.65,
                                      cols: int = 5,
                                      img_size: float = 2.0,
                                      **plot_kwargs):
        """
        Display the concepts attribution maps for the images given in argument.

        Parameters
        ----------
        images
            The images to display.
        importances
            The importances computed by the estimate_importance() method.
            If None is provided, then the global importances will be used, otherwise
            the local importances set in this parameter will be used.
        nb_most_important_concepts
            The number of concepts to focus on. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            It is applied after the filter_percentile operation.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        cols
            Number of columns. Default to 3.
        img_size
            Size of each subplots (in inch), considering we keep aspect ratio.
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """

        self.plot_concept_attribution_legend(nb_most_important_concepts=nb_most_important_concepts)

        # Take into account single vs multiple images array
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        # Configure the subplots
        layout_configuration, rows, figwidth, figheight = \
            self.compute_subplots_layout_parameters(images=images, cols=cols, img_size=img_size)

        fig = plt.figure()
        fig.set_size_inches(figwidth, figheight)
        fig.subplots_adjust(**layout_configuration)

        # If importances is passed in argument, then use these local importances,
        # otherwise use the global importances stored in self.importances
        if importances is None:
            # global
            most_important_concepts = self.sensitivity.most_important_concepts
        else:
            # local
            most_important_concepts = np.argsort(importances)[::-1]

        most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        for i, img in enumerate(images):
            plt.subplot(rows, cols, i+1)

            self.plot_concept_attribution_map(image=img,
                                              most_important_concepts=most_important_concepts,
                                              filter_percentile=filter_percentile,
                                              clip_percentile=clip_percentile,
                                              alpha=alpha,
                                              **plot_kwargs)

    def plot_concept_attribution_map(self,
                                    image: np.ndarray,
                                    most_important_concepts: np.ndarray,
                                    nb_most_important_concepts: int = 5,
                                    filter_percentile: int = 90,
                                    clip_percentile: Optional[float] = 10,
                                    alpha: float = 0.65,
                                    **plot_kwargs):
        """
        Display the concepts attribution map for a single image given in argument.

        Parameters
        ----------
        image
            The image to display.
        most_important_concepts
            The concepts ids to display.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap.
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            It is applied after the filter_percentile operation.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        # pylint: disable=E1101
        most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        # Find the colors corresponding to the importances
        global_color_index_order = np.argsort(self.sensitivity.importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        local_cmap = np.array(self.sensitivity.cmaps)[local_color_index_order]

        if image.shape[0] == 3:
            dsize = image.shape[1:3] # pytorch
        else:
            dsize = image.shape[0:2] # tf
        BaseCraft._show(image, **plot_kwargs)

        image_u = self.transform(image)[0]
        for i, c_id in enumerate(most_important_concepts[::-1]):
            heatmap = image_u[:, :, c_id]

            # only show concept if excess N-th percentile
            sigma = np.percentile(np.array(heatmap).flatten(), filter_percentile)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            # resize the heatmap before cliping
            heatmap = cv2.resize(heatmap[:, :, None], dsize=dsize,
                                 interpolation=cv2.INTER_CUBIC)
            if clip_percentile:
                heatmap = _clip_percentile(heatmap, clip_percentile)

            BaseCraft._show(heatmap, cmap=local_cmap[::-1][i], alpha=alpha, **plot_kwargs)

    def plot_image_concepts(self,
                            img: np.ndarray,
                            display_importance_order: DisplayImportancesOrder = \
                                                    DisplayImportancesOrder.GLOBAL,
                            nb_most_important_concepts: int = 5,
                            filter_percentile: int = 90,
                            clip_percentile: Optional[float] = 10,
                            alpha: float = 0.65,
                            filepath: Optional[str] = None,
                            **plot_kwargs):
        """
        All in one method displaying several plots for the image `id` given in argument:
        - the concepts attribution map for this image
        - the best crops for each concept (displayed around the heatmap)
        - the importance of each concept

        Parameters
        ----------
        img
            The image to display.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
            Default to GLOBAL.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
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
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        fig = plt.figure(figsize=(18, 7))

        if display_importance_order == DisplayImportancesOrder.LOCAL:
            # compute the importances for the sample input in argument
            importances = self.estimate_importance(inputs=img)
            most_important_concepts = np.argsort(importances)[::-1][:nb_most_important_concepts]
        else:
            # use the global importances computed on the whole dataset
            importances = self.sensitivity.importances
            most_important_concepts = \
                self.sensitivity.most_important_concepts[:nb_most_important_concepts]

        # create the main gridspec which is split in the left and right parts storing
        # the crops, and the central part to display the heatmap
        nb_rows = ceil(len(most_important_concepts) / 2.0)
        nb_cols = 4
        gs_main = fig.add_gridspec(nb_rows, nb_cols, hspace=0.4, width_ratios=[0.2, 0.4, 0.2, 0.4])

        # Central image
        #
        fig.add_subplot(gs_main[:, 1])
        self.plot_concept_attribution_map(image=img,
                                          most_important_concepts=most_important_concepts,
                                          nb_most_important_concepts=nb_most_important_concepts,
                                          filter_percentile=filter_percentile,
                                          clip_percentile=clip_percentile,
                                          alpha=alpha,
                                          **plot_kwargs)

        # Concepts: creation of the axes on left and right of the image for the concepts
        #
        gs_concepts_axes  = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 0])
                             for i in range(nb_rows)]
        gs_right = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 2])
                    for i in range(nb_rows)]
        gs_concepts_axes.extend(gs_right)

        # display the best crops for each concept, in the order of the most important concept
        nb_crops = 6

        # compute the right color to use for the crops
        global_color_index_order = np.argsort(self.sensitivity.importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        local_cmap = np.array(self.sensitivity.cmaps)[local_color_index_order]

        for i, c_id in enumerate(most_important_concepts):
            cmap = local_cmap[i]

            # use a ghost invisible subplot only to have a border around the crops
            ghost_axe = fig.add_subplot(gs_concepts_axes[i][:,:])
            ghost_axe.set_title(f"{c_id}", color=cmap(1.0))
            ghost_axe.axis('off')

            inset_axes = ghost_axe.inset_axes([-0.04, -0.04, 1.08, 1.08]) # outer border
            inset_axes.set_xticks([])
            inset_axes.set_yticks([])
            for spine in inset_axes.spines.values(): # border color
                spine.set_edgecolor(color=cmap(1.0))
                spine.set_linewidth(3)

            # draw each crop for this concept
            gs_current = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=
                                                                gs_concepts_axes[i][:,:])

            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]
            for i in range(nb_crops):
                axe = plt.Subplot(fig, gs_current[i // 3, i % 3])
                fig.add_subplot(axe)
                BaseCraft._show(best_crops[i])

        # Right plot: importances
        importance_axe  = gridspec.GridSpecFromSubplotSpec(3, 2, width_ratios=[0.1, 0.9],
                                                                 height_ratios=[0.15, 0.6, 0.15],
                                                                 subplot_spec=gs_main[:, 3])
        fig.add_subplot(importance_axe[1, 1])
        self.plot_concepts_importances(importances=importances,
                                       display_importance_order=display_importance_order,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()
