from math import ceil
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt

from xplique.plots.image import _clip_percentile
from .craft import Factorization, Sensitivity
from .holistic_craft_object_detection import HolisticCraftObjectDetection, show_ax
from .latent_extractor import LatentData, LatentExtractor
from ..attributions import GradientInput
from ..utils_functions.object_detection.tf.box_manager import filter_boxes as filter_boxes_tf

class HolisticCraftObjectDetectionTf(HolisticCraftObjectDetection):

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20,
    ):
        super().__init__(latent_extractor, number_of_concepts)
        self.cmaps = [
            Sensitivity._get_alpha_cmap(cmap) for cmap in plt.get_cmap("tab10").colors
        ]
        if number_of_concepts > 10:
            print("warning: increase cmaps to match new number of concepts !")

    # def filter_boxes_tf(model_output_result: list, class_id: int, accuracy: float=0.55):
    #     all_filtered = []
    #     for nbc_annotations in model_output_result:
    #         probas = nbc_annotations[:, 5:]
    #         class_ids = tf.argmax(probas, axis=1)
    #         scores = nbc_annotations[:, 4]
    #         keep = tf.logical_and(tf.equal(class_ids, class_id), scores > accuracy)
    #         print("Number of kept annotations:", tf.reduce_sum(tf.cast(keep, tf.int32)).numpy())
    #         # print(f"idx of keep: {tf.where(keep).numpy()}")
    #         filtered_boxes = tf.boolean_mask(nbc_annotations, keep)
    #         # print(f"boxes: {filtered_boxes[:, :5]}")
    #         all_filtered.append(filtered_boxes)
    #     return all_filtered

    def fit(
        self, inputs: tf.Tensor, class_id: int = 0, max_iter: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # pass the data through the 1st part of the model
        # latent_data = self.latent_extractor.latent_predict(inputs) # ok
        # latent_data = self.latent_extractor.input_to_latent(inputs) # ko
        latent_data = self.latent_extractor.input_to_latent_batched(inputs)
        # get the activations on CPU
        activations = latent_data.get_activations()
        activations_original_shape = activations.shape[:-1]
        activations_flat = np.reshape(activations.numpy(), (-1, activations.shape[-1])) # specific to tf

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


    def transform(self, inputs: tf.Tensor, resize=None) -> np.ndarray:
        # latent_data = self.model_wrapper.latent_predict(inputs, resize)
        latent_data = self.latent_extractor.input_to_latent_batched(inputs, resize)
        coeffs_u = self.transform_latent(latent_data)
        return coeffs_u

    def transform_latent(self, latent_data: LatentData) -> np.ndarray:
        self.check_if_fitted()
        activations = latent_data.get_activations()
        activations = activations.numpy() # TF
        activations_original_shape = activations.shape[:-1]
        activations_flat = np.reshape(activations, (-1, activations.shape[-1]))
        print(f"[HolisticCraftObjectDetectionTf] activations_flat shape: {activations_flat.shape} from original activations shape: {activations.shape}")

        # compute coeffs_u using the NMF
        w_dtype = self.factorization.reducer.components_.dtype
        coeffs_u = self.factorization.reducer.transform(
            activations_flat.astype(w_dtype)
            #self._to_np_array(activations_flat, dtype=w_dtype)
        )
        coeffs_u = np.reshape(coeffs_u, (*activations_original_shape, -1))
        print(f"[HolisticCraftObjectDetectionTf] final coeffs_u shape: {coeffs_u.shape} from original activations shape: {activations_original_shape},  -1")
        return coeffs_u

    def encode(self, inputs: tf.Tensor, resize=None) -> np.ndarray:
        latent_data = self.latent_extractor.input_to_latent_batched(inputs, resize)
        coeffs_u = self.transform_latent(latent_data)
        return latent_data, coeffs_u

    def decode(self, latent_data: LatentData, coeffs_u: np.ndarray) -> tf.Tensor:
        # warning: coeffs_u can be only a part of the coeffs of the latent_data, when
        # using an operator with a batch size > 1 (batched operator will pass parts of
        # coeffs_u to the decoder)
        self.check_if_fitted()
        # TODO : batchify
        activations = coeffs_u @ self.factorization.concept_bank_w
        # rest of the model
        activations = tf.convert_to_tensor(activations, dtype=tf.float32)

        # create activations
        latent_data.set_activations(activations)
        return self.latent_extractor.latent_to_logit_batched(latent_data)
    
    def decode_tf(self, latent_data: LatentData, coeffs_u: tf.Tensor) -> tf.Tensor:
        # warning: coeffs_u can be only a part of the coeffs of the latent_data, when
        # using an operator with a batch size > 1 (batched operator will pass parts of
        # coeffs_u to the decoder)
        self.check_if_fitted()
        print(f"[HolisticCraftObjectDetectionTf] decode_tf: coeffs_u.shape={coeffs_u.shape}, concept_bank_w.shape={self.factorization.concept_bank_w.shape}")
        activations = coeffs_u @ tf.convert_to_tensor(self.factorization.concept_bank_w, dtype=tf.float32)
        # rest of the model

        # create activations
        latent_data.set_activations(activations)
        return self.latent_extractor.latent_to_logit_batched(latent_data)

    def make_decoder(self, latent_data: LatentData) -> tf.keras.Model:
        parent_craft = self
        
        class ConceptToLogitModel(tf.keras.layers.Layer):
            def __init__(self, latent_data, **kwargs):
                super().__init__(**kwargs)
                self.latent_data = latent_data
            
            def set_latent_data(self, latent_data: LatentData):
                self.latent_data = latent_data

            def call(self, coeffs_u):
                return parent_craft.decode_tf(self.latent_data, coeffs_u)
        
        return ConceptToLogitModel(latent_data)
  

    def display_images_per_concept(
        self,
        images,
        filter_percentile=80,
        clip_percentile=5,
        order=None,
    ):
        nb_cols = self.number_of_concepts
        nb_rows = len(images)
        image_shape = images[0].shape
        if image_shape[0] == 3:  # channels first (rare en TF)
            dsize = image_shape[1:3]
        else:  # channels last (standard TF)
            dsize = image_shape[0:2]
        dsize = (dsize[1], dsize[0])

        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
        axs = np.atleast_2d(axs) # fix issue when nb_rows == 1

        if order is None:
            concepts_id = list(range(self.number_of_concepts))
        else:
            concepts_id = order

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        # a voir si on peut pas retourner un np tout le temps pour factoriser toute cette methode entre tf et torch:
        coeffs_u = self.transform(images)#.detach().cpu().numpy()

        for i, c_i in enumerate(concepts_id):
            for image_id, image in enumerate(images):
                # show_ax(image.squeeze().permute(1, 2, 0).numpy(), ax=axs[image_id, i])
                show_ax(image.numpy(), ax=axs[image_id, i])
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

    def compute_gradient_input(self, images, operator, class_id, accuracy=0.9, harmonize=True, batch_size=1):
        # need to create a decoder for each batch of images because the latent data and the images tensor
        # must be the same size
        # batch_size = self.latent_extractor.batch_size
        print(f"[HolisticCraftObjectDetectionTf] compute_gradient_input: batch_size={batch_size}, images.shape={images.shape}, class_id={class_id}, accuracy={accuracy}")
        nb_batchs = ceil(len(images) / batch_size)
        start_ids = [i * batch_size for i in range(nb_batchs)]

        explanation_list = []
        for i in start_ids:
            a_batch = images[i : i + batch_size]
            print(f"[HolisticCraftObjectDetectionTf] processing batch {i} with shape {a_batch.shape}")

            latent_data, coeffs_u = self.encode(a_batch)
            decoder = self.make_decoder(latent_data) # activate gradients

            # filter classes
            print(f"[HolisticCraftObjectDetectionTf] Calling the decoder with coeffs_u.shape={coeffs_u.shape}")
            output = decoder(tf.convert_to_tensor(coeffs_u))
            # class_filtered_output = filter_boxes_same_dim_tf(output, class_id=class_id, accuracy=accuracy)
            class_filtered_output = filter_boxes_tf(output, class_id=class_id, accuracy=accuracy)
            for j, boxes in enumerate(class_filtered_output):
                if len(boxes) == 0:
                    print(f"[HolisticCraftObjectDetectionTf] No boxes found for class_id={class_id} with accuracy={accuracy} on image {i+j}.")
                    continue
                stacked_np = tf.convert_to_tensor(boxes)[tf.newaxis, ...]  # add batch dimension
                print(f"[HolisticCraftObjectDetectionTf] stacked_np shape: {stacked_np.shape}")
                print(f"[HolisticCraftObjectDetectionTf] using the GradientInput explainer with operator={operator}, harmonize={harmonize}, batch_size={batch_size}")
                explainer = GradientInput(decoder, operator=operator, harmonize=harmonize, batch_size=1)
                explanation = explainer.explain(inputs=coeffs_u, targets=stacked_np)
                explanation_list.append(explanation)
        
        print(f"[HolisticCraftObjectDetectionTf] explanation_list length: {len(explanation_list)}")
        if len(explanation_list) == 0:
            raise ValueError("No explanations were generated. Check if the images contain boxes for the specified class_id and accuracy.")
        explanation = tf.concat(explanation_list, axis=0)
        return explanation

    def estimate_importance_gradient_input_xplique(self, images, operator, class_id, accuracy=0.9, batch_size=1):
        explanation = self.compute_gradient_input(images, operator, class_id, accuracy=accuracy, harmonize=False, batch_size=batch_size)
        # sum the explanation over the batch and spatial dimensions
        # importances = np.sum(explanation, axis=(0, 1, 2))
        print(f"[HolisticCraftObjectDetectionTf] estimate_importance_gradient_input_xplique: explanation.shape={explanation.shape}")
        sum_per_image_and_concept = np.sum(explanation, axis=(1,2))
        importances = np.mean(sum_per_image_and_concept, axis=0) # mean over the images
        # TODO: normalize & abs
        return importances
    
