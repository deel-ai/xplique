"""Shared eager execution for whole-concept-channel interventions."""

from numbers import Integral

import numpy as np
import tensorflow as tf

from ...commons import batch_tensor, repeat_labels
from ...types import Callable, OperatorSignature, Optional, Union
from ..base import BlackBoxExplainer, sanitize_input_output


class _ConceptChannelExplainer(BlackBoxExplainer):
    """Private orchestration for input-shaped, globally broadcast concept effects."""

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        seed: int = 0,
    ):
        if batch_size is not None and (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, Integral)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, Integral)
            or not -(2**63) <= seed < 2**63
        ):
            raise ValueError("seed must be a signed 64-bit integer.")
        super().__init__(model, None if batch_size is None else int(batch_size), operator)
        self.seed = int(seed)

    @sanitize_input_output
    def explain(self, inputs: tf.Tensor, targets: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Explain finite, channel-last coefficients with fixed targets.

        Parameters
        ----------
        inputs
            Dense coefficients of shape (N, ..., K), or a paired dataset passed
            with targets=None. Support is determined after float32 sanitization.
        targets
            Fixed targets with the same batch length as inputs.

        Returns
        -------
        explanations
            Float32 tensor with the input shape. Each channel effect is broadcast
            across positions, not distributed over them. Inactive channels are zero.

        Raises
        ------
        ValueError
            If inputs, target batches, or scalar operator outputs are invalid.
        """
        if not tf.executing_eagerly():
            raise ValueError("Concept-channel explanations require eager execution.")
        if inputs.shape.rank < 2 or inputs.shape[-1] == 0:
            raise ValueError("inputs must have rank at least two and a nonempty concept axis.")
        if targets.shape.rank < 1 or inputs.shape[0] != targets.shape[0]:
            raise ValueError("targets must have a batch axis matching inputs.")
        if not bool(tf.reduce_all(tf.math.is_finite(inputs))):
            raise ValueError("inputs must contain only finite coefficients.")
        if inputs.shape[0] == 0:
            return tf.zeros_like(inputs)

        explanations = []
        for input_index, single_input in enumerate(inputs):
            active = tf.reduce_any(single_input != 0, axis=tf.range(tf.rank(single_input) - 1))
            active_ids = tf.where(active)[:, 0]
            nb_active = int(tf.size(active_ids))
            if nb_active == 0:
                explanations.append(tf.zeros_like(single_input))
                continue

            masks = self._sample_masks(nb_active, input_index)
            outputs = []
            batch_size = self.batch_size or int(masks.shape[0])
            for mask_batch in batch_tensor(masks, batch_size):
                size = int(mask_batch.shape[0])
                channel_masks = tf.transpose(
                    tf.scatter_nd(
                        active_ids[:, None],
                        tf.transpose(mask_batch),
                        [single_input.shape[-1], size],
                    )
                )
                broadcast_shape = (
                    [size] + [1] * (single_input.shape.rank - 1) + [single_input.shape[-1]]
                )
                perturbed = single_input[None] * tf.reshape(channel_masks, broadcast_shape)
                repeated_targets = repeat_labels(targets[input_index : input_index + 1], size)
                scores = tf.convert_to_tensor(
                    self.inference_function(self.model, perturbed, repeated_targets)
                )
                if scores.shape not in (tf.TensorShape([size]), tf.TensorShape([size, 1])):
                    raise ValueError(
                        "operator must return one scalar per perturbation: (B,) or (B, 1)."
                    )
                scores = tf.cast(tf.reshape(scores, [size]), tf.float64)
                if not bool(tf.reduce_all(tf.math.is_finite(scores))):
                    raise ValueError("operator must return finite scalar scores.")
                outputs.append(scores)

            effects = self._estimate(masks, tf.concat(outputs, axis=0))
            ambient = tf.scatter_nd(active_ids[:, None], effects, [single_input.shape[-1]])
            explanations.append(tf.broadcast_to(ambient, tf.shape(single_input)))
        return tf.stack(explanations)

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        raise NotImplementedError

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
