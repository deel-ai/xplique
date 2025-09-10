from math import ceil
from typing import Tuple

import copy
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt

from xplique.plots.image import _clip_percentile
from .craft import Factorization, Sensitivity
from .holistic_craft_object_detection import HolisticCraftObjectDetection, show_ax
from .latent_extractor import LatentData, LatentExtractor
from ..attributions import GradientInput
from ..utils_functions.object_detection.torch.box_manager import filter_boxes, filter_boxes_same_dim
from xplique.wrappers import TorchWrapper

# from xplique.attributions.global_sensitivity_analysis import (
#     HaltonSequenceRS,
#     JansenEstimator,
# )


# from sklearn.metrics import mean_squared_error
# from xplique.wrappers import TorchWrapper
# from ..attributions import SobolAttributionMethod



class HolisticCraftObjectDetectionTorch(HolisticCraftObjectDetection):

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20,
        device: str = "cuda",
    ):
        super().__init__(latent_extractor, number_of_concepts)
        self.device = device
        
        self.cmaps = [
            Sensitivity._get_alpha_cmap(cmap) for cmap in plt.get_cmap("tab10").colors
        ]
        if number_of_concepts > 10:
            print("warning: increase cmaps to match new number of concepts !")

    def fit(
        self, inputs: torch.tensor, class_id: int = 0, max_iter: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # pass the data through the 1st part of the model
        # latent_data = self.latent_extractor.latent_predict(inputs) # ok
        # latent_data = self.latent_extractor.input_to_latent(inputs) # ko
        latent_data = self.latent_extractor.input_to_latent_batched(inputs)
        # get the activations on CPU
        activations = latent_data.get_activations()
        activations_original_shape = activations.shape[:-1]
        activations_flat = np.reshape(activations.detach().cpu().numpy(), (-1, activations.shape[-1]))

        # apply NMF to the activations to obtain matrices U and W
        reducer = NMF(
            n_components=self.number_of_concepts, alpha_W=1e-2, max_iter=max_iter
        )

        print(f"activations shape before NMF: {activations.shape}")
        coeffs_u = reducer.fit_transform(activations_flat)
        print(f"coeffs_u shape after NMF.fit_transform(): {coeffs_u.shape}")
        coeffs_u = np.reshape(coeffs_u, (*activations_original_shape, -1))
        print(f"coeffs_u rearranged shape: {coeffs_u.shape}")
        concept_bank_w = reducer.components_.astype(np.float32)

        self.factorization = Factorization(
            inputs, class_id, latent_data, reducer, coeffs_u, concept_bank_w
        )

        return None, activations, coeffs_u, concept_bank_w

    def _to_np_array(self, inputs: torch.Tensor, dtype: type = None):
        """
        Converts a Pytorch tensor into a numpy array.
        """
        res = inputs.detach().cpu().numpy()
        if dtype is not None:
            return res.astype(dtype)
        return res

    def transform(self, inputs: torch.Tensor, resize=None) -> np.ndarray:
        # latent_data = self.model_wrapper.latent_predict(inputs, resize)
        latent_data = self.latent_extractor.input_to_latent_batched(inputs, resize)
        coeffs_u = self.transform_latent(latent_data)
        return coeffs_u

    def encode(self, inputs: torch.Tensor, resize=None) -> np.ndarray:
        latent_data = self.latent_extractor.input_to_latent_batched(inputs, resize)
        coeffs_u = self.transform_latent(latent_data)
        return latent_data, coeffs_u

    def decode(self, latent_data: LatentData, coeffs_u: np.ndarray) -> torch.Tensor:
        # warning: coeffs_u can be only a part of the coeffs of the latent_data, when
        # using an operator with a batch size > 1 (batched operator will pass parts of
        # coeffs_u to the decoder)
        self.check_if_fitted()
        # TODO : batchify
        activations = coeffs_u @ self.factorization.concept_bank_w
        # rest of the model
        activations = torch.tensor(activations, dtype=torch.float32, device=self.device)

        # create activations
        latent_data.set_activations(activations)
        return self.latent_extractor.latent_to_logit_batched(latent_data)
        # return self.latent_extractor.latent_to_logit(latent_data)

    def decode_torch(self, latent_data: LatentData, coeffs_u: torch.Tensor, no_grad=True) -> torch.Tensor:
        # warning: coeffs_u can be only a part of the coeffs of the latent_data, when
        # using an operator with a batch size > 1 (batched operator will pass parts of
        # coeffs_u to the decoder)
        self.check_if_fitted()
        activations = coeffs_u @ torch.tensor(self.factorization.concept_bank_w, device=coeffs_u.device)
        # rest of the model
        # activations = torch.tensor(activations, dtype=torch.float32, device=self.device)
        activations = activations.to(dtype=torch.float32, device=self.device)

        # create activations
        latent_data.set_activations(activations)
        return self.latent_extractor.latent_to_logit_batched(latent_data, no_grad)
        # return self.latent_extractor.latent_to_logit(latent_data)
    
    def make_decoder(self, latent_data: LatentData, no_grad=True) -> nn.Module:
        parent_craft = self
        class ConceptToLogitModel(torch.nn.Module):
            def __init__(self, latent_data: LatentData, no_grad):
                super().__init__()
                self.latent_data = latent_data
                self.no_grad = no_grad

            def set_no_grad(self, no_grad: bool):
                self.no_grad = no_grad

            def set_latent_data(self, latent_data: LatentData):
                self.latent_data = latent_data
            
            def forward(self, coeffs_u: torch.Tensor) -> torch.Tensor:
                return parent_craft.decode_torch(self.latent_data, coeffs_u, self.no_grad)

            ## non on ne peut pas faire de TorchWrapper() si un latent est passe en params, il ne faut que des tensors
            # def forward(self, latent_data: LatentData, coeffs_u: torch.Tensor) -> torch.Tensor:
            #     return parent_craft.decode_torch(latent_data, coeffs_u)
            
        return ConceptToLogitModel(latent_data, no_grad)

    def transform_latent(self, latent_data: LatentData) -> np.ndarray:
        self.check_if_fitted()
        activations = latent_data.get_activations()
        activations = activations.detach().cpu().numpy()
        activations_original_shape = activations.shape[:-1]
        activations_flat = np.reshape(activations, (-1, activations.shape[-1]))

        # compute coeffs_u using the NMF
        w_dtype = self.factorization.reducer.components_.dtype
        coeffs_u = self.factorization.reducer.transform(
            activations_flat.astype(w_dtype)
            #self._to_np_array(activations_flat, dtype=w_dtype)
        )
        coeffs_u = np.reshape(coeffs_u, (*activations_original_shape, -1))
        return coeffs_u

    def coeffs_u_to_activations(self, coeffs_u: np.ndarray):
        self.check_if_fitted()
        activations = coeffs_u @ self.factorization.concept_bank_w
        return activations

    # def display_concepts_for_image(
    #     self, image, filter_percentile=90, clip_percentile=2, nb_rows=2, order=None
    # ):
    #     if image.squeeze().shape[0] == 3:
    #         dsize = image.squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = image.squeeze().shape[0:2]  # tf

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     fig, axs = plt.subplots(
    #         nb_rows, self.number_of_concepts // nb_rows, figsize=(25, 5 * nb_rows)
    #     )
    #     axes = axs.flatten()

    #     coeffs_u = self.transform(image)
    #     for i, ax in enumerate(axes):
    #         show_ax(image.squeeze().permute(1, 2, 0).detach().numpy(), ax=ax)
    #         c_i = concepts_id[i]
    #         concept_heatmap = coeffs_u[0, :, :, c_i]

    #         # only show concept if excess N-th percentile
    #         sigma = np.percentile(
    #             np.array(concept_heatmap).flatten(), filter_percentile
    #         )
    #         heatmap = concept_heatmap * np.array(concept_heatmap > sigma, np.float32)

    #         # resize the heatmap before cliping
    #         heatmap = cv2.resize(
    #             heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #         )
    #         heatmap = _clip_percentile(heatmap, clip_percentile)

    #         show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.75, ax=ax)
    #         ax.axis("off")

    # def display_concepts_for_image_torch(
    #     self, image, filter_percentile=90, clip_percentile=2, nb_rows=2, order=None
    # ):
    #     if image.squeeze().shape[0] == 3:
    #         dsize = image.squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = image.squeeze().shape[0:2]  # tf

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     fig, axs = plt.subplots(
    #         nb_rows, self.number_of_concepts // nb_rows, figsize=(25, 5 * nb_rows)
    #     )
    #     axes = axs.flatten()

    #     coeffs_u = self.transform_torch(image)
    #     for i, ax in enumerate(axes):
    #         show_ax(image.squeeze().permute(1, 2, 0).numpy(), ax=ax)
    #         c_i = concepts_id[i]
    #         concept_heatmap = np.array(
    #             coeffs_u[0, :, :, c_i].cpu().detach(), np.float32
    #         )

    #         # only show concept if excess N-th percentile
    #         sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #         heatmap = concept_heatmap * (concept_heatmap > sigma)

    #         # resize the heatmap before cliping
    #         heatmap = cv2.resize(
    #             heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #         )
    #         heatmap = _clip_percentile(heatmap, clip_percentile)

    #         # show_ax(heatmap, cmap=self.cmaps[::-1][c_i], alpha=0.75, ax=ax) # test mais ko
    #         show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.75, ax=ax)
    #         ax.axis("off")

    # def display_concepts_for_all_images(
    #     self, images, filter_percentile=90, clip_percentile=2
    # ):
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf

    #     nb_rows = len(images)
    #     fig, axs = plt.subplots(
    #         nb_rows, self.number_of_concepts, figsize=(25, 2 * nb_rows)
    #     )
    #     axs = np.atleast_2d(axs)

    #     for image_id in range(nb_rows):

    #         coeffs_u = self.transform(images[image_id])
    #         for i in range(self.number_of_concepts):

    #             show_ax(
    #                 images[image_id].squeeze().permute(1, 2, 0).numpy(),
    #                 ax=axs[image_id, i],
    #             )

    #             concept_heatmap = coeffs_u[0, :, :, i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(
    #                 np.array(concept_heatmap).flatten(), filter_percentile
    #             )
    #             heatmap = concept_heatmap * np.array(
    #                 concept_heatmap > sigma, np.float32
    #             )

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)

    #             # show(heatmap, alpha=0.1)
    #             show_ax(
    #                 heatmap, cmap=self.cmaps[::-1][i], alpha=0.75, ax=axs[image_id, i]
    #             )

    #             axs[image_id, i].axis("off")

    # def display_concepts_for_all_images_torch(
    #     self, images, filter_percentile=90, clip_percentile=2, order=None
    # ):
    #     nb_cols = self.number_of_concepts
    #     nb_rows = len(images)
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf

    #     fig, axs = plt.subplots(
    #         nb_rows, self.number_of_concepts, figsize=(25, 2 * nb_rows)
    #     )
    #     axs = np.atleast_2d(axs)

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order
    #     # coeffs_u = self.factorization.coeffs_u.detach().numpy()
    #     # coeffs_u = self.transform_torch(images)
    #     for j, image in enumerate(images):

    #         for i, c_i in enumerate(concepts_id):
    #             coeffs_u = self.transform_torch(image)
    #             show_ax(image.squeeze().permute(1, 2, 0).numpy(), ax=axs[j, i])
    #             # print(f"{images_preprocessed[image_id].squeeze().shape = }")
    #             # concept_heatmap = coeffs_u[j, :, :, i].detach().numpy()
    #             concept_heatmap = coeffs_u[0, :, :, c_i].detach().numpy()

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #             heatmap = concept_heatmap * (concept_heatmap > sigma)

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)

    #             # show(heatmap, alpha=0.1)
    #             show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.5, ax=axs[j, i])
    #     return fig

    # def display_top_images_per_concept(
    #     self,
    #     images,
    #     nb_top_images=3,
    #     filter_percentile=90,
    #     clip_percentile=2,
    #     order=None,
    # ):
    #     nb_cols = self.number_of_concepts
    #     nb_rows = nb_top_images
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf

    #     fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
    #     axs = np.atleast_2d(axs)

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     for i, c_i in enumerate(concepts_id):
    #         axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

    #     # coeffs_u = self.transform(images)
    #     coeffs_u = self.factorization.coeffs_u

    #     for i, c_i in enumerate(concepts_id):

    #         top_image_for_concept_i = np.argsort(np.mean(coeffs_u, (1, 2))[:, c_i])[
    #             ::-1
    #         ]

    #         for j, image_id in enumerate(top_image_for_concept_i[:nb_top_images]):
    #             show_ax(
    #                 images[image_id].squeeze().permute(1, 2, 0).detach().numpy(),
    #                 ax=axs[j, i],
    #             )
    #             # print(f"{images_preprocessed[image_id].squeeze().shape = }")
    #             concept_heatmap = coeffs_u[image_id, :, :, c_i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(
    #                 np.array(concept_heatmap).flatten(), filter_percentile
    #             )
    #             heatmap = concept_heatmap * np.array(
    #                 concept_heatmap > sigma, np.float32
    #             )

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)

    #             # show(heatmap, alpha=0.1)
    #             show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.5, ax=axs[j, i])

    # def display_top_images_per_concept_torch(
    #     self,
    #     images,
    #     nb_top_images=3,
    #     filter_percentile=80,
    #     clip_percentile=5,
    #     order=None,
    # ):
    #     nb_cols = self.number_of_concepts
    #     nb_rows = nb_top_images
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf

    #     fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
    #     axs = np.atleast_2d(axs)

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     for i, c_i in enumerate(concepts_id):
    #         axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

    #     # coeffs_u = self.transform(images)
    #     # coeffs_u = self.factorization.coeffs_u.cpu().detach().numpy()
    #     coeffs_u = self.transform_torch(images).cpu().detach().numpy()
    #     print(f"{coeffs_u.shape = }")

    #     for i, c_i in enumerate(concepts_id):
    #         top_image_for_concept_i = np.argsort(np.mean(coeffs_u, (1, 2))[:, c_i])[
    #             ::-1
    #         ]

    #         for j, image_id in enumerate(top_image_for_concept_i[:nb_top_images]):
    #             show_ax(
    #                 images[image_id].squeeze().permute(1, 2, 0).numpy(), ax=axs[j, i]
    #             )
    #             # print(f"{images_preprocessed[image_id].squeeze().shape = }")
    #             concept_heatmap = coeffs_u[image_id, :, :, c_i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #             heatmap = concept_heatmap * (concept_heatmap > sigma)

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)

    #             # show(heatmap, alpha=0.1)
    #             show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.5, ax=axs[j, i])
    #     return fig


    # todo: remove part about tf, and put it in HolisticCraftObjectDetectionTf
    def display_images_per_concept(
        self,
        images,
        filter_percentile=80,
        clip_percentile=5,
        order=None,
    ):
        nb_cols = self.number_of_concepts
        nb_rows = len(images)
        if images[0].squeeze().shape[0] == 3:
            dsize = images[0].squeeze().shape[1:3]  # pytorch
        else:
            dsize = images[0].squeeze().shape[0:2]  # tf
        dsize = (dsize[1], dsize[0])  # cv2 expects (width, height)

        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
        axs = np.atleast_2d(axs) # fix issue when nb_rows == 1

        if order is None:
            concepts_id = list(range(self.number_of_concepts))
        else:
            concepts_id = order

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        coeffs_u = self.transform(images)
        # coeffs_u = coeffs_u.detach().cpu().numpy() # specific to torch

        for i, c_i in enumerate(concepts_id):
            for image_id, image in enumerate(images):
                show_ax(image.squeeze().permute(1, 2, 0).detach().cpu().numpy(), ax=axs[image_id, i])
                concept_heatmap = coeffs_u[image_id, :, :, c_i]

                # only show concept if excess N-th percentile
                sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
                heatmap = concept_heatmap * (concept_heatmap > sigma)

                # resize the heatmap before cliping
                heatmap = cv2.resize(
                    heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
                )
                heatmap = _clip_percentile(heatmap, clip_percentile)
                
                # show(heatmap, alpha=0.1)
                show_ax(
                    heatmap, cmap=self.cmaps[::-1][c_i], alpha=0.5, ax=axs[image_id, i]
                )
        return fig
    
    # def display_images_per_concept_torch(
    #     self,
    #     images,
    #     filter_percentile=80,
    #     clip_percentile=5,
    #     order=None,
    # ):
    #     nb_cols = self.number_of_concepts
    #     nb_rows = len(images)
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf
    #     dsize = (dsize[1], dsize[0])  # cv2 expects (width, height)

    #     fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
    #     axs = np.atleast_2d(axs) # fix issue when nb_rows == 1

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     for i, c_i in enumerate(concepts_id):
    #         axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

    #     # coeffs_u = self.transform(images)
    #     # coeffs_u = self.factorization.coeffs_u.cpu().detach().numpy()
    #     coeffs_u = self.transform_torchnmf(images).cpu().detach().numpy()

    #     for i, c_i in enumerate(concepts_id):
    #         for image_id, image in enumerate(images):
    #             show_ax(image.squeeze().permute(1, 2, 0).numpy(), ax=axs[image_id, i])
    #             concept_heatmap = coeffs_u[image_id, :, :, c_i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #             heatmap = concept_heatmap * (concept_heatmap > sigma)

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)
                
    #             # show(heatmap, alpha=0.1)
    #             show_ax(
    #                 heatmap, cmap=self.cmaps[::-1][i], alpha=0.5, ax=axs[image_id, i]
    #             )
    #     return fig

    # def display_top_images_per_concept_with_feature_viz(
    #     self,
    #     images,
    #     feature_viz_images,
    #     nb_top_images=3,
    #     filter_percentile=90,
    #     clip_percentile=2,
    #     order=None,
    # ):
    #     nb_cols = self.number_of_concepts
    #     nb_rows = nb_top_images + 2  # 2 features viz
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf

    #     fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(35, 3 * nb_rows), dpi=300)
    #     axs = np.atleast_2d(axs)

    #     if order is None:
    #         concepts_id = list(range(self.number_of_concepts))
    #     else:
    #         concepts_id = order

    #     for i, c_i in enumerate(concepts_id):
    #         axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

    #     # coeffs_u = self.transform(images)
    #     coeffs_u = self.factorization.coeffs_u.cpu().detach().numpy()
    #     print(f"{coeffs_u.shape = }")
    #     for i, c_i in enumerate(concepts_id):
    #         # print(f"display_top_images_per_concept_with_feature_viz, {i = }, {c_i = }")
    #         top_image_for_concept_i = np.argsort(np.mean(coeffs_u, (1, 2))[:, c_i])[
    #             ::-1
    #         ]

    #         for j, image_id in enumerate(top_image_for_concept_i[:nb_top_images]):
    #             show_ax(
    #                 images[image_id].squeeze().permute(1, 2, 0).numpy(), ax=axs[j, i]
    #             )
    #             # print(f"{images_preprocessed[image_id].squeeze().shape = }")
    #             concept_heatmap = coeffs_u[image_id, :, :, c_i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #             heatmap = concept_heatmap * (concept_heatmap > sigma)

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)

    #             # show(heatmap, alpha=0.1)
    #             show_ax(heatmap, cmap=self.cmaps[::-1][i], alpha=0.5, ax=axs[j, i])

    #         try:
    #             axs[3, i].imshow(feature_viz_images[i][0])  # maco
    #             axs[4, i].imshow(feature_viz_images[i][1])  # fourier
    #         except:
    #             pass
    #         axs[3, i].axis("off")
    #         axs[4, i].axis("off")
    #     return fig


    # def get_concepts_heatmaps(
    #     self,
    #     images,
    #     filter_percentile=80,
    #     clip_percentile=5
    # ):
    #     if images[0].squeeze().shape[0] == 3:
    #         dsize = images[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = images[0].squeeze().shape[0:2]  # tf
    #     dsize = (dsize[1], dsize[0])  # cv2 expects (width, height)

    #     coeffs_u = self.transform(images)#.detach().cpu().numpy()

    #     # heatmaps = np.empty(
    #     #     (len(images), self.number_of_concepts, dsize[1], dsize[0])
    #     # )
        
    #     heatmaps = np.empty(
    #         (len(images), dsize[1], dsize[0], self.number_of_concepts)
    #     )
        
    #     for c_i in range(self.number_of_concepts):
    #         for image_id, image in enumerate(images):
    #             concept_heatmap = coeffs_u[image_id, :, :, c_i]

    #             # only show concept if excess N-th percentile
    #             sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
    #             heatmap = concept_heatmap * (concept_heatmap > sigma)

    #             # resize the heatmap before cliping
    #             heatmap = cv2.resize(
    #                 heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
    #             )
    #             heatmap = _clip_percentile(heatmap, clip_percentile)
    #             # heatmaps[image_id][c_i] = heatmap
    #             heatmaps[image_id,:,:,c_i] = heatmap
    #     return heatmaps


    # def filter_boxes_torch(self, output_result: list, class_id: int, accuracy: float=0.55):
    #     filtered = []
    #     for nbc_annotations in output_result:
    #         probas = nbc_annotations[:, 5:]
    #         class_ids = probas.argmax(dim=1)
    #         scores = nbc_annotations[:, 4]
    #         keep = (class_ids == class_id) & (scores > accuracy)
    #         print("Number of kept annotations:", keep.sum().item())
    #         print(f"idx of keep: {np.where(keep.cpu().numpy())[0]}")
    #         print(f"boxes: {nbc_annotations[keep, :5]}")
    #         filtered.append(nbc_annotations[keep, :])
    #     return filtered

    # def filter_boxes_tf(self, model_output_result: list, class_id: int, accuracy: float=0.55):
    #     import tensorflow as tf
    #     all_filtered = []
    #     for nbc_annotations in model_output_result:
    #         probas = nbc_annotations[:, 5:]
    #         class_ids = tf.argmax(probas, axis=1)
    #         scores = nbc_annotations[:, 4]
    #         keep = tf.logical_and(tf.equal(class_ids, class_id), scores > accuracy)
    #         # print("Number of kept annotations:", tf.reduce_sum(tf.cast(keep, tf.int32)).numpy())
    #         # print(f"idx of keep: {tf.where(keep).numpy()}")
    #         filtered_boxes = tf.boolean_mask(nbc_annotations, keep)
    #         # print(f"boxes: {filtered_boxes[:, :5]}")
    #         all_filtered.append(filtered_boxes)
    #     return all_filtered
    
    # def compute_gradient_input_per_class(self, torch_inputs: torch.Tensor, class_id: int, device: torch.device, batch_size=1):
    #     from xplique.wrappers import TorchWrapper
    #     from xplique.attributions import GradientInput
    #     from xplique import Tasks
    #     import tensorflow as tf

    #     torch_wrapped_model = TorchWrapper(self.latent_extractor.eval(), device=device, is_channel_first=True)

    #     inputs = torch_inputs.permute(0, 2, 3, 1)
    #     targets = torch_wrapped_model(inputs)

    #     filtered_targets = self.filter_boxes_tf(targets, class_id=class_id, accuracy=0.9)
    #     # print(f"Filtered targets[0] shape: {filtered_targets[0].shape}")
        
    #     explanation_list = []
    #     for idx_image_to_explain in range(len(torch_inputs)):
    #         nb_boxes_to_explain = filtered_targets[idx_image_to_explain].shape[0]
    #         # print(f"{nb_boxes_to_explain = }")
            
    #         # print(f"{filtered_targets[idx_image_to_explain].shape = }")
    #         explainer = GradientInput(torch_wrapped_model, operator=Tasks.OBJECT_DETECTION, batch_size=batch_size)

    #         explanation = explainer.explain(inputs[idx_image_to_explain:idx_image_to_explain + 1], filtered_targets[idx_image_to_explain][tf.newaxis,:,:])
    #         explanation_list.append(explanation)

    #     return explanation_list

    # def estimate_importance_gradient_input(self, images, class_id, filter_percentile=80, clip_percentile=5):
    #     import tensorflow as tf

    #     # heatmaps shape: (N, H, W, 1)
    #     heatmaps = self.get_concepts_heatmaps(images, filter_percentile, clip_percentile)
        
    #     # explanations shape: (N, H, W, 1)
    #     explanation_list = self.compute_gradient_input_per_class(images, class_id, device=self.device)
    #     explanations = tf.concat(explanation_list, axis=0)

    #     res = heatmaps * explanations
    #     importances_values = np.array([tf.reduce_sum(res[:, :, :, c_i]).numpy() for c_i in range(self.number_of_concepts)])
    #     importances = importances_values.argsort()
    #     return importances


    # def estimate_importance_deletion_xplique(self, images, operator, class_id, accuracy=0.9):

    #     # def filter_boxes(output_result: list, class_id: int, accuracy: float=0.55):
    #     #     filtered = []
    #     #     for nbc_annotations in output_result:
    #     #         probas = nbc_annotations[:, 5:]
    #     #         class_ids = probas.argmax(dim=1)
    #     #         scores = nbc_annotations[:, 4]
    #     #         keep = (class_ids == class_id) & (scores > accuracy)
    #     #         filtered.append(nbc_annotations[keep, :])
    #     #     return filtered

    #     latent_data, coeffs_u = self.encode(images)
    #     decoder = self.make_decoder(latent_data)
    #     torch_wrapped_decoder = TorchWrapper(decoder.eval(), device=self.device, is_channel_first=False)

    #     output = decoder(coeffs_u)

    #     # filter classes
    #     class_filtered_output = filter_boxes(output, class_id=class_id, accuracy=accuracy)
    #     class_filtered_output = tf.stack([tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output], axis=0)

    #     # compute score for only 1 class
    #     tf_coeffs_u = torch.Tensor(coeffs_u)
    #     scores_without_modification = operator(model=torch_wrapped_decoder, inputs=tf_coeffs_u, targets=class_filtered_output)
    #     scores_without_modification = scores_without_modification.numpy().mean()
    #     print(f"{scores_without_modification = }")

    #     # remove each concept in U and compute the predictions
    #     importances = []
    #     for i in range(self.number_of_concepts):
    #         coeffs_u_modified = torch.Tensor(np.copy(coeffs_u))
    #         coeffs_u_modified[:, :, :, i:i+1] = 0

    #         score_only_persons = operator(model=torch_wrapped_decoder, inputs=coeffs_u_modified, targets=class_filtered_output)
    #         importances.append(score_only_persons.numpy().mean())
    #     importances = np.array(importances)

    #     return importances

    def compute_gradient_input(self, images, operator, class_id, accuracy=0.9, harmonize=True, batch_size=1):
        # need to create a decoder for each batch of images because the latent data and the images tensor
        # must be the same size
        # batch_size = self.latent_extractor.batch_size
        nb_batchs = ceil(len(images) / batch_size)
        start_ids = [i * batch_size for i in range(nb_batchs)]

        explanation_list = []
        for i in start_ids:
            a_batch = images[i : i + batch_size]

            latent_data, coeffs_u = self.encode(a_batch)
            decoder = self.make_decoder(latent_data) # activate gradients

            # filter classes
            output = decoder(torch.tensor(coeffs_u, device=self.device))
            class_filtered_output = filter_boxes_same_dim(output, class_id=class_id, accuracy=accuracy)
            # class_filtered_output = tf.stack([tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output], axis=0)
            # class_filtered_output = [tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output]
            
            # @TODO: iterer sur les filtres comme en tf (i.e. une liste plutot qu'un array)
            np_arrays = [t.detach().cpu().numpy() for t in class_filtered_output]
            stacked_np = np.stack(np_arrays, axis=0)
            # class_filtered_output_tf = tf.convert_to_tensor(stacked_np)

            # check size of the dimensions
            # _ = decoder(torch.tensor(coeffs_u))
            # _ = torch_wrapped_decoder(coeffs_u)
            decoder.set_no_grad(False)  # set no_grad to True to avoid computing gradients in the decoder (gradients needed for WhiteBox)
            torch_wrapped_decoder = TorchWrapper(decoder.eval(), device=self.device, is_channel_first=False)
            explainer = GradientInput(torch_wrapped_decoder, operator=operator, harmonize=harmonize, batch_size=batch_size)
            explanation = explainer.explain(inputs=coeffs_u, targets=stacked_np)
            explanation_list.append(explanation)
            
        explanation = tf.concat(explanation_list, axis=0)
        return explanation

    # def compute_sobol(self, images, operator, explainer_class, class_id, accuracy=0.9, harmonize=True, batch_size=1, nb_design=32, grid_size=1):
    #     # need to create a decoder for each batch of images because the latent data and the images tensor
    #     # must be the same size
    #     # batch_size = self.latent_extractor.batch_size
    #     nb_batchs = ceil(len(images) / batch_size)
    #     start_ids = [i * batch_size for i in range(nb_batchs)]

    #     explanation_list = []
    #     for i in start_ids:
    #         a_batch = images[i : i + batch_size]
    #         effective_batch_size = a_batch.shape[0]

    #         latent_data, coeffs_u = self.encode(a_batch)
    #         decoder = self.make_decoder(latent_data) # activate gradients

    #         # filter classes
    #         output = decoder(torch.tensor(coeffs_u, device=self.device))
    #         class_filtered_output = filter_boxes_same_dim(output, class_id=class_id, accuracy=accuracy)
    #         # class_filtered_output = tf.stack([tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output], axis=0)
    #         # class_filtered_output = [tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output]
            
    #         np_arrays = [t.detach().cpu().numpy() for t in class_filtered_output]
    #         stacked_np = np.stack(np_arrays, axis=0)

    #         # check size of the dimensions
    #         # _ = decoder(torch.tensor(coeffs_u))
    #         # _ = torch_wrapped_decoder(coeffs_u)
    #         decoder.set_no_grad(True)  # set no_grad to True to avoid computing gradients in the decoder (BlackBox)
    #         torch_wrapped_decoder = TorchWrapper(decoder.eval(), device=self.device, is_channel_first=False)
    #         explainer = explainer_class(torch_wrapped_decoder, operator=operator, grid_size=grid_size, nb_channels=self.number_of_concepts, nb_design=nb_design, batch_size=effective_batch_size)
    #         explanation = explainer.explain(inputs=coeffs_u, targets=stacked_np)
    #         explanation_list.append(explanation)
            
    #     explanation = tf.concat(explanation_list, axis=0)
    #     return explanation

    def estimate_importance_gradient_input_xplique(self, images, operator, class_id, accuracy=0.9, batch_size=1):
        explanation = self.compute_gradient_input(images, operator, class_id, accuracy=accuracy, harmonize=False, batch_size=batch_size)
        # sum the explanation over the batch and spatial dimensions
        # importances = np.sum(explanation, axis=(0, 1, 2))
        sum_per_image_and_concept = np.sum(explanation, axis=(1,2))
        importances = np.mean(sum_per_image_and_concept, axis=0) # mean over the images
        # TODO: normalize & abs
        return importances


    # def estimate_importance_sobol_xplique(self, images, operator, class_id, accuracy=0.9, batch_size=1):
    #     grid_size = 1
    #     explainer_class = SobolAttributionMethod
    #     explanation = self.compute_sobol(images, operator, explainer_class, class_id, accuracy=accuracy, harmonize=False, batch_size=batch_size, grid_size=grid_size)
    #     importances = np.mean(explanation, (0, 1, 2))
    #     return importances


    # def estimate_importance_gradient_input_xplique_old(self, images, operator, class_id, accuracy=0.9):
    #     from xplique.wrappers import TorchWrapper
    #     import tensorflow as tf
    #     from ..attributions import GradientInput

    #     def filter_boxes(output_result: list, class_id: int, accuracy: float=0.55):
    #         filtered = []
    #         for nbc_annotations in output_result:
    #             probas = nbc_annotations[:, 5:]
    #             class_ids = probas.argmax(dim=1)
    #             scores = nbc_annotations[:, 4]
    #             keep = (class_ids == class_id) & (scores > accuracy)
    #             filtered.append(nbc_annotations[keep, :])
    #         return filtered

    #     latent_data, coeffs_u = self.encode(images)
    #     decoder = self.make_decoder(latent_data)
    #     torch_wrapped_decoder = TorchWrapper(decoder.eval(), device=self.device, is_channel_first=False)

    #     # output = decoder(coeffs_u)
    #     output = decoder(torch.tensor(coeffs_u, device=self.device))

    #     # filter classes
    #     class_filtered_output = filter_boxes(output, class_id=class_id, accuracy=accuracy)
    #     class_filtered_output = tf.stack([tf.constant(torch_tensor.detach().cpu().numpy()) for torch_tensor in class_filtered_output], axis=0)

    #     # compute score for only 1 class
    #     tf_coeffs_u = torch.Tensor(coeffs_u)
    #     explainer = GradientInput(torch_wrapped_decoder, operator=operator, harmonize=False, batch_size=2)
    #     explanation = explainer.explain(inputs=tf_coeffs_u, targets=class_filtered_output)

    #     # sum the explanation over the batch and spatial dimensions
    #     importances = np.sum(explanation, axis=(0, 1, 2))
    #     return importances

        
    # def estimate_importance(
    #     self,
    #     inputs: np.ndarray = None,
    #     nb_design: int = 32,
    #     force_mask=None,
    #     force_mask_others=None,
    #     force_bank=None,
    #     force_bank_others=None,
    #     nb_images_limit=None,
    # ) -> np.ndarray:

    #     U = (
    #         torch.from_numpy(self.factorization.coeffs_u)
    #         .to(torch.float32)
    #         .to(device="cpu")
    #     )
    #     W = torch.from_numpy(self.factorization.concept_bank_w).to(device="cpu")
    #     A = U @ W

    #     import copy

    #     latent_data_reconstructed = copy.deepcopy(self.factorization.latent_data)
    #     latent_data_reconstructed.set_activations(A)

    #     scores_reconstructed = self.model_wrapper.logit_predict(
    #         latent_data=latent_data_reconstructed, class_id=25
    #     )
    #     # outputs_reconstructed = model.h(latent_data_reconstructed[0].to(device))
    #     print("scores sans perturbations:", scores_reconstructed)

    #     from xplique.attributions.global_sensitivity_analysis import (
    #         HaltonSequenceRS,
    #         JansenEstimator,
    #     )

    #     masks = HaltonSequenceRS()(self.number_of_concepts, nb_design=32)
    #     estimator = JansenEstimator()

    #     W = torch.from_numpy(self.factorization.concept_bank_w).to(device="cpu")

    #     importances = []
    #     # coeffs_u : (N, W, H, R) with R = nb_concepts
    #     for coeff in U:
    #         # print(f"{coeff.shape = }")
    #         u_perturbated = coeff[None, :] * masks[:, None, None, :]
    #         # print(f"{u_perturbated.shape = }")

    #         a_perturbated = np.reshape(u_perturbated, (-1, coeff.shape[-1])) @ W
    #         a_perturbated = np.reshape(
    #             a_perturbated, (len(masks), U.shape[1], U.shape[2], -1)
    #         )
    #         # print(f"{a_perturbated.shape = }")
    #         # A = A[:16]

    #         scores_for_img = []

    #         batch_size = len(U)
    #         nb_batchs = ceil(len(a_perturbated) / batch_size)
    #         start_ids = [i * batch_size for i in range(nb_batchs)]

    #         for i in start_ids:
    #             a_batch = a_perturbated[i : i + batch_size]
    #             latent_data_reconstructed[0].set_activations(a_batch)
    #             # latent_data_reconstructed[0].pdim()

    #             score_reconstructed = self.model_wrapper.logit_predict(
    #                 latent_data=latent_data_reconstructed, class_id=25
    #             )
    #             scores_for_img.extend(score_reconstructed)
    #         # print("scores avec perturbations:", scores_for_img)
    #         stis = estimator(masks, scores_for_img, 32)
    #         importances.append(stis)

    #     importances = np.mean(importances, 0)

    #     print(importances)

    # # def estimate_importance_deletion(self, class_id, top_k=3):
    # #     with torch.no_grad():
    # #         # compute original predictions
    # #         original_predictions = self.model_wrapper.logit_predict(latent_data=self.factorization.latent_data, class_id=class_id, top_k=top_k)
    # #         # print(f"{original_predictions = }")
    # #         # print(f"ori pred for image 0: {original_predictions[0:1]}")

    # #         # delete each concept in U and compute the predictions
    # #         W = self.factorization.concept_bank_w.T.detach().clone().to(device='cpu')
    # #         mse_deletion = []
    # #         for c_i in range(self.number_of_concepts):
    # #             latent_data_with_deletion = copy.deepcopy(self.factorization.latent_data)
    # #             U = self.factorization.coeffs_u.detach().clone().to(device='cpu')
    # #             U[:,:,:,c_i] = 0 # delete concept c_i
    # #             A = U @ W
    # #             latent_data_with_deletion.set_activations(A)
    # #             current_predictions = self.model_wrapper.logit_predict(latent_data=latent_data_with_deletion, class_id=class_id, top_k=top_k)
    # #             # if c_i == 9:
    # #             # print(f"pred for concept {c_i} image 0: {current_predictions[0]}")
    # #             # compute mse between original predictions and the current prediction
    # #             # mse_deletion.append(mean_squared_error(original_predictions[0:1], current_predictions[0:1]))
    # #             mse_deletion.append(mean_squared_error(original_predictions, current_predictions))
    # #         importances = mse_deletion
    # #         return importances

    # def estimate_importance_deletion(self, class_id, top_k=3):
    #     with torch.no_grad():
    #         # compute original predictions
    #         # original_predictions = self.model_wrapper.output_predict(latent_data=self.factorization.latent_data) #, class_id=class_id, top_k=top_k)

    #         # probas = torch.stack([op.get_probas().clone().detach() for op in original_predictions])
    #         # probas_for_our_class = probas[:,:,class_id]
    #         probas_for_our_class = self.model_wrapper.probas_predict(
    #             latent_data=self.factorization.latent_data, class_id=class_id
    #         )
    #         original_scores = probas_for_our_class.sort(descending=True)[0][
    #             :, :top_k
    #         ].mean(1)

    #         # delete each concept in U and compute the predictions
    #         W = self.factorization.concept_bank_w.T.detach().clone().to(device="cpu")
    #         mse_deletion = []
    #         for c_i in range(self.number_of_concepts):
    #             latent_data_with_deletion = copy.deepcopy(
    #                 self.factorization.latent_data
    #             )
    #             U = self.factorization.coeffs_u.detach().clone().to(device="cpu")
    #             U[:, :, :, c_i] = 0  # delete concept c_i
    #             A = U @ W
    #             latent_data_with_deletion.set_activations(A)
    #             # current_predictions = self.model_wrapper.output_predict(latent_data=latent_data_with_deletion) #s, class_id=class_id, top_k=top_k)
    #             # probas = torch.stack([op.get_probas().clone().detach() for op in current_predictions])
    #             # probas_for_our_class = probas[:,:,class_id]
    #             probas_for_our_class = self.model_wrapper.probas_predict(
    #                 latent_data=latent_data_with_deletion, class_id=class_id
    #             )
    #             current_scores = probas_for_our_class.sort(descending=True)[0][
    #                 :, :top_k
    #             ].mean(1)

    #             mse_deletion.append(
    #                 mean_squared_error(original_scores.cpu(), current_scores.cpu())
    #             )
    #         importances = mse_deletion
    #         return importances

    # def estimate_importance_deletion2(self, class_id, top_k=3):
    #     with torch.no_grad():
    #         # compute original predictions
    #         original_scores = self.model_wrapper.predict_score_for_class(
    #             latent_data=self.factorization.latent_data,
    #             class_id=class_id,
    #             top_k=top_k,
    #         )

    #         # probas = torch.stack([op.get_probas().clone().detach() for op in original_predictions])
    #         # probas_for_our_class = probas[:,:,class_id]
    #         # original_scores = probas_for_our_class.sort(descending=True)[0][:,:top_k].mean(1)

    #         # delete each concept in U and compute the predictions
    #         W = self.factorization.concept_bank_w.T.detach().clone().to(device="cpu")
    #         mse_deletion = []
    #         for c_i in range(self.number_of_concepts):
    #             latent_data_with_deletion = copy.deepcopy(
    #                 self.factorization.latent_data
    #             )
    #             U = self.factorization.coeffs_u.detach().clone().to(device="cpu")
    #             U[:, :, :, c_i] = 0  # delete concept c_i
    #             A = U @ W
    #             latent_data_with_deletion.set_activations(A)
    #             current_scores = self.model_wrapper.predict_score_for_class(
    #                 latent_data=latent_data_with_deletion,
    #                 class_id=class_id,
    #                 top_k=top_k,
    #             )
    #             # probas = torch.stack([op.get_probas().clone().detach() for op in current_predictions])
    #             # probas_for_our_class = probas[:,:,class_id]
    #             # current_scores = probas_for_our_class.sort(descending=True)[0][:,:top_k].mean(1)

    #             mse_deletion.append(
    #                 mean_squared_error(original_scores.cpu(), current_scores.cpu())
    #             )
    #         importances = mse_deletion
    #         return importances

    # def estimate_importance_sobol(self, class_id, top_k=3):
    #     masks = HaltonSequenceRS()(self.number_of_concepts, nb_design=32)
    #     estimator = JansenEstimator()

    #     with torch.no_grad():
    #         # compute original predictions
    #         original_predictions = self.model_wrapper.logit_predict(
    #             latent_data=self.factorization.latent_data,
    #             class_id=class_id,
    #             top_k=top_k,
    #         )
    #         print(f"ori pred for image 0: {original_predictions[0:1]}")

    #         W = self.factorization.concept_bank_w.T.detach().clone().to(device="cpu")
    #         U = self.factorization.coeffs_u.detach().clone().to(device="cpu")
    #         importances = []
    #         # coeffs_u : (N, W, H, R) with R = nb_concepts
    #         for coeff in U:
    #             latent_data_reconstructed = copy.deepcopy(
    #                 self.factorization.latent_data
    #             )
    #             u_perturbated = coeff[None, :] * masks[:, None, None, :]
    #             # a_perturbated = np.reshape(u_perturbated,
    #             #                         (-1, coeff.shape[-1])) @ W
    #             # a_perturbated = np.reshape(a_perturbated,
    #             #                         (len(masks), U.shape[1], U.shape[2], -1))
    #             a_perturbated = u_perturbated @ W

    #             batch_size = len(U)
    #             nb_batchs = ceil(len(a_perturbated) / batch_size)
    #             start_ids = [i * batch_size for i in range(nb_batchs)]

    #             scores_for_img = []
    #             for i in start_ids:
    #                 a_batch = a_perturbated[i : i + batch_size]
    #                 latent_data_reconstructed.set_activations(a_batch)
    #                 score_reconstructed = self.model_wrapper.logit_predict(
    #                     latent_data_reconstructed, class_id=class_id, top_k=top_k
    #                 )
    #                 scores_for_img.extend(score_reconstructed)
    #             stis = estimator(masks, scores_for_img, 32)
    #             importances.append(stis)

    #         importances = np.mean(importances, 0)
    #         return importances

    # def estimate_importance_sobol2(self, class_id, top_k=3):
    #     masks = HaltonSequenceRS()(self.number_of_concepts, nb_design=32)
    #     estimator = JansenEstimator()

    #     with torch.no_grad():
    #         # compute original predictions
    #         # compute original predictions, the top_k prediction and their indices

    #         # original_probas_for_our_class = self._probas_predict(latent_data=self.factorization.latent_data, class_id=class_id)
    #         # idx_top_k = torch.argsort(original_probas_for_our_class, descending=True, dim=1)[:, 0:top_k] # n_images x top_k indices sorted
    #         # row_indices = torch.arange(len(idx_top_k)).unsqueeze(1).expand(-1, top_k)
    #         # # original_scores_top_k = original_probas_for_our_class[row_indices, idx_top_k]

    #         original_predictions = self.model_wrapper.output_predict(
    #             latent_data=self.factorization.latent_data
    #         )
    #         probas = torch.stack(
    #             [op.get_probas().clone().detach() for op in original_predictions]
    #         )
    #         original_probas_for_our_class = probas[:, :, class_id]
    #         idx_top_k = torch.argsort(
    #             original_probas_for_our_class, descending=True, dim=1
    #         )[
    #             :, 0:top_k
    #         ]  # n_images x top_k indices sorted
    #         row_indices = torch.arange(len(idx_top_k)).unsqueeze(1).expand(-1, top_k)

    #         W = self.factorization.concept_bank_w.T.detach().clone().to(device="cpu")
    #         U = self.factorization.coeffs_u.detach().clone().to(device="cpu")
    #         importances = []
    #         # coeffs_u : (N, W, H, R) with R = nb_concepts
    #         for coeff in U:
    #             latent_data_reconstructed = copy.deepcopy(
    #                 self.factorization.latent_data
    #             )
    #             u_perturbated = coeff[None, :] * masks[:, None, None, :]
    #             # a_perturbated = np.reshape(u_perturbated,
    #             #                         (-1, coeff.shape[-1])) @ W
    #             # a_perturbated = np.reshape(a_perturbated,
    #             #                         (len(masks), U.shape[1], U.shape[2], -1))
    #             a_perturbated = u_perturbated @ W

    #             batch_size = len(U)
    #             nb_batchs = ceil(len(a_perturbated) / batch_size)
    #             start_ids = [i * batch_size for i in range(nb_batchs)]

    #             scores_for_img = []
    #             for i in start_ids:
    #                 a_batch = a_perturbated[i : i + batch_size]
    #                 latent_data_reconstructed.set_activations(a_batch)
    #                 # current_probas = self._probas_predict(latent_data=latent_data_reconstructed, class_id=class_id)
    #                 # current_scores_top_k = current_probas[row_indices, idx_top_k].cpu().numpy()

    #                 current_predictions = self.model_wrapper.output_predict(
    #                     latent_data=self.factorization.latent_data
    #                 )
    #                 probas = torch.stack(
    #                     [op.get_probas().clone().detach() for op in current_predictions]
    #                 )
    #                 current_probas_for_our_class = probas[:, :, class_id]
    #                 current_scores_top_k = (
    #                     current_probas_for_our_class[row_indices, idx_top_k]
    #                     .cpu()
    #                     .numpy()
    #                 )

    #                 scores_for_img.extend(current_scores_top_k)
    #             stis = estimator(masks, scores_for_img, 32)
    #             importances.append(stis)

    #         importances = np.mean(importances, 0)
    #         return importances

    # def get_accumulated_gradients(self, inputs, class_id):

    #     # inputs: (N, C, H, W)
    #     if len(inputs.shape) == 3:
    #         inputs = inputs.unsqueeze(0)  # Add an extra dim for a single image

    #     # Batch preparation
    #     nb_batchs = ceil(len(inputs) / self.batch_size)
    #     start_ids = [i * self.batch_size for i in range(nb_batchs)]
    #     accumulated_gradients = []

    #     for i in start_ids:
    #         i_end = min(i + self.batch_size, len(inputs))
    #         batch = inputs[i:i_end].to(self.device)
    #         batch.requires_grad_()

    #         # mini-batch inference
    #         latent_data = self.model_wrapper.input_to_latent_model(batch)
    #         probas_for_our_class = self.model_wrapper.probas_predict(
    #             latent_data, class_id, no_grad=False
    #         )
    #         score = probas_for_our_class.mean()
    #         score.backward()

    #         # Retrieve gradients of the current batch
    #         gradients = (
    #             batch.grad.permute(0, 2, 3, 1).detach().cpu().numpy()
    #         )  # Fix memory issues by moving to cpu
    #         accumulated_gradients.append(gradients)

    #         # Cleanup
    #         del batch, latent_data, probas_for_our_class, score
    #         torch.cuda.empty_cache()

    #     # Gradients accumulation management
    #     accumulated_gradients = np.concatenate(accumulated_gradients, axis=0)
    #     print(f"Gradients shape: {accumulated_gradients.shape}")
    #     return accumulated_gradients

    # def estimate_importance_input_x_gradient(self, inputs, class_id):
    #     if inputs[0].squeeze().shape[0] == 3:
    #         dsize = inputs[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = inputs[0].squeeze().shape[0:2]  # tf
    #     print(f"{dsize = }")

    #     # inputs: (N, C, H, W)
    #     accumulated_gradients = self.get_accumulated_gradients(inputs, class_id)

    #     # compute gradients * U for each image and each concept
    #     input_x_gradient_results = np.zeros(
    #         (len(accumulated_gradients), self.number_of_concepts)
    #     )
    #     for i in range(len(accumulated_gradients)):
    #         gradients_image = accumulated_gradients[i]  # 800, 800, 3
    #         heatmap = (
    #             self.factorization.coeffs_u[i].detach().cpu().numpy()
    #         )  # 25, 25, 10 # TODO: recompute U instead of using the one from the factorization

    #         for c_i in range(self.number_of_concepts):
    #             concept_heatmap = heatmap[:, :, c_i]
    #             concept_heatmap = cv2.resize(
    #                 np.repeat(concept_heatmap[:, :, None], 3, axis=-1),
    #                 dsize=dsize,
    #                 interpolation=cv2.INTER_CUBIC,
    #             )
    #             input_x_gradient = gradients_image * concept_heatmap
    #             input_x_gradient_results[i, c_i] = input_x_gradient.sum()

    #     return input_x_gradient_results.mean(0)

    # # Pour Fcos
    # def estimate_importance_input_x_gradient2(self, inputs, class_id):
    #     if inputs[0].squeeze().shape[0] == 3:
    #         dsize = inputs[0].squeeze().shape[1:3]  # pytorch
    #     else:
    #         dsize = inputs[0].squeeze().shape[0:2]  # tf
    #     print(f"{dsize = }")

    #     # inputs: (N, C, H, W)
    #     if len(inputs.shape) == 3:
    #         inputs = inputs.unsqueeze(0)  # Add an extra dim for a single image

    #     # Batch preparation
    #     nb_batchs = ceil(len(inputs) / self.batch_size)
    #     start_ids = [i * self.batch_size for i in range(nb_batchs)]
    #     accumulated_gradients = []

    #     for i in start_ids:
    #         i_end = min(i + self.batch_size, len(inputs))
    #         batch = inputs[i:i_end].to(self.device)
    #         batch.requires_grad_()

    #         # mini-batch inference
    #         latent_data = self.model_wrapper.input_to_latent_model(batch)

    #         score = self.model_wrapper.predict_score_for_class(
    #             latent_data, class_id, top_k=3, no_grad=False
    #         )
    #         score.backward()

    #         # Retrieve gradients of the current batch
    #         gradients = batch.grad.permute(0, 2, 3, 1).detach().cpu().numpy()
    #         # gradients = batch.grad.detach().cpu().numpy()  # Fix memory issues by moving to cpu
    #         accumulated_gradients.append(gradients)

    #         # Cleanup
    #         del batch, latent_data, probas_for_our_class, score
    #         torch.cuda.empty_cache()

    #     # Gradients accumulation management
    #     accumulated_gradients = np.concatenate(accumulated_gradients, axis=0)
    #     print(f"Gradients shape: {accumulated_gradients.shape}")

    #     # compute gradients * U for each image and each concept
    #     input_x_gradient_results = np.zeros(
    #         (len(accumulated_gradients), self.number_of_concepts)
    #     )
    #     for i in range(len(accumulated_gradients)):
    #         gradients_image = accumulated_gradients[i]  # 800, 800, 3
    #         heatmap = (
    #             self.factorization.coeffs_u[i].detach().cpu().numpy()
    #         )  # 25, 25, 10

    #         for c_i in range(self.number_of_concepts):
    #             concept_heatmap = heatmap[:, :, c_i]
    #             concept_heatmap = cv2.resize(
    #                 np.repeat(concept_heatmap[:, :, None], 3, axis=-1),
    #                 dsize=dsize,
    #                 interpolation=cv2.INTER_CUBIC,
    #             )
    #             input_x_gradient = gradients_image * concept_heatmap
    #             input_x_gradient_results[i, c_i] = input_x_gradient.sum()

    #     return input_x_gradient_results.mean(0)
