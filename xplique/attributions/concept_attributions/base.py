"""Shared eager execution for whole-concept-channel interventions."""

from numbers import Integral

import numpy as np
import tensorflow as tf

from ...commons import batch_tensor, repeat_labels, tensor_sanitize
from ...types import Any, Callable, List, OperatorSignature, Optional, Tuple, Union
from ..base import BlackBoxExplainer, sanitize_input_output
from .interactions import ConceptInteractionResult, _resolve_pairs, _validate_pairs


def _check_integer(
    value, message: str, minimum: int, maximum: Optional[int] = None, even: bool = False
) -> int:
    """Return value as int if it is a non-boolean integer in [minimum, maximum), else raise."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
        or (maximum is not None and value >= maximum)
        or (even and value % 2)
    ):
        raise ValueError(message)
    return int(value)


def _bernoulli(shape: List[int], seed: tf.Tensor) -> tf.Tensor:
    """Draw stateless float32 Bernoulli(1/2) masks."""
    uniform = tf.random.stateless_uniform(shape, seed=seed, dtype=tf.float32)
    return tf.cast(uniform < 0.5, tf.float32)


class _ConceptChannelExplainer(BlackBoxExplainer):
    """Private orchestration for input-shaped, globally broadcast concept effects.

    Subclasses define the mask design (_sample_masks) and the singleton estimator
    (_estimate). Setting _supports_interactions and implementing
    _estimate_pair_chunk (optionally _prepare_interactions and _main_effects)
    enables explain_interactions from the same per-input mask design.
    """

    _supports_interactions = False

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        seed: int = 0,
    ):
        if batch_size is not None:
            batch_size = _check_integer(
                batch_size, "batch_size must be a positive integer or None.", 1
            )
        seed = _check_integer(seed, "seed must be a signed 64-bit integer.", -(2**63), 2**63)
        super().__init__(model, batch_size, operator)
        self.seed = seed

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
        self._validate_inputs_targets(inputs, targets)
        if inputs.shape[0] == 0:
            return tf.zeros_like(inputs)

        explanations = []
        for input_index, single_input in enumerate(inputs):
            active_ids, masks, outputs = self._evaluate_input(
                single_input, targets[input_index : input_index + 1], input_index
            )
            if masks is None:
                explanations.append(tf.zeros_like(single_input))
                continue
            effects = self._estimate(masks, outputs)
            ambient = tf.scatter_nd(active_ids[:, None], effects, [single_input.shape[-1]])
            explanations.append(tf.broadcast_to(ambient, tf.shape(single_input)))
        return tf.stack(explanations)

    def explain_interactions(
        self,
        inputs,
        targets=None,
        *,
        pairs=None,
        pair_batch_size=256,
    ) -> List[ConceptInteractionResult]:
        """Estimate singleton and pairwise scores from one mask design per input.

        Parameters
        ----------
        inputs
            Dense channel-last coefficients (N, ..., K) or a paired dataset.
        targets
            Fixed targets for the inputs; omit only for a paired dataset.
        pairs
            Optional integer (P, 2) ambient concept indices, ordered i < j.
            None evaluates all distinct active pairs. Inactive requested pairs
            have exact zero scores; unrequested pairs are not evaluated.
        pair_batch_size
            Maximum number of pair features evaluated together, independently
            of the inference batch size.

        Returns
        -------
        results
            One indexed ConceptInteractionResult per input, with no
            spatial axes. Score semantics are documented by each explainer.
            An empty batch returns [].

        Raises
        ------
        NotImplementedError
            If the explainer does not support interactions.
        ValueError
            If the sanitized inputs, targets, pairs or pair batch size are invalid.
        """
        if not self._supports_interactions:
            raise NotImplementedError(
                f"{type(self).__name__} does not support explain_interactions()."
            )
        pair_batch_size = _check_integer(
            pair_batch_size, "pair_batch_size must be a positive integer.", 1
        )
        inputs, targets = tensor_sanitize(inputs, targets)
        self._validate_inputs_targets(inputs, targets)
        n_concepts = int(inputs.shape[-1])
        requested_pairs = _validate_pairs(pairs, n_concepts)

        results = []
        for input_index, single_input in enumerate(inputs):
            # Reuse one mask design and one per-input state for all scores.
            active_ids, masks, outputs = self._evaluate_input(
                single_input, targets[input_index : input_index + 1], input_index
            )
            ambient_pairs, local_pairs, valid_rows = _resolve_pairs(
                active_ids.numpy(), requested_pairs, n_concepts
            )
            pair_scores = np.zeros(len(ambient_pairs), np.float32)
            if masks is None:
                main_effects = tf.zeros([0], tf.float32)
            else:
                state = self._prepare_interactions(masks, outputs)
                main_effects = self._main_effects(state)
                # Score active pairs in bounded feature chunks; inactive rows stay zero.
                for start in range(0, len(valid_rows), pair_batch_size):
                    rows = valid_rows[start : start + pair_batch_size]
                    pair_scores[rows] = self._estimate_pair_chunk(
                        state, tf.convert_to_tensor(local_pairs[rows])
                    ).numpy()
            results.append(
                ConceptInteractionResult(
                    n_concepts,
                    active_ids,
                    main_effects,
                    tf.convert_to_tensor(ambient_pairs, tf.int64),
                    tf.convert_to_tensor(pair_scores),
                )
            )
        return results

    @staticmethod
    def _validate_inputs_targets(inputs: tf.Tensor, targets: tf.Tensor) -> None:
        if not tf.executing_eagerly():
            raise ValueError("Concept-channel explanations require eager execution.")
        if inputs.shape.rank < 2 or inputs.shape[-1] == 0:
            raise ValueError("inputs must have rank at least two and a nonempty concept axis.")
        if targets.shape.rank < 1 or inputs.shape[0] != targets.shape[0]:
            raise ValueError("targets must have a batch axis matching inputs.")
        if not bool(tf.reduce_all(tf.math.is_finite(inputs))):
            raise ValueError("inputs must contain only finite coefficients.")

    @staticmethod
    def _active_channel_ids(single_input: tf.Tensor) -> tf.Tensor:
        active = tf.reduce_any(single_input != 0, axis=tf.range(tf.rank(single_input) - 1))
        return tf.where(active)[:, 0]

    def _input_seed(self, input_index: int) -> tf.Tensor:
        """Fold the original input index into the stateless explainer seed."""
        return tf.random.experimental.stateless_fold_in(
            tf.constant([self.seed, 0], dtype=tf.int64), tf.cast(input_index, tf.int64)
        )

    def _evaluate_input(
        self, single_input: tf.Tensor, single_target: tf.Tensor, input_index: int
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Sample and evaluate one input's design; masks and outputs are None without support."""
        active_ids = self._active_channel_ids(single_input)
        nb_active = int(tf.size(active_ids))
        if nb_active == 0:
            return active_ids, None, None
        masks = self._sample_masks(nb_active, input_index)
        outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
        return active_ids, masks, outputs

    def _evaluate_masks(
        self,
        single_input: tf.Tensor,
        single_target: tf.Tensor,
        active_ids: tf.Tensor,
        masks: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate a mask design once, in inference batches, returning float64 scores."""
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
            repeated_targets = repeat_labels(single_target, size)
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
        return tf.concat(outputs, axis=0)

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        raise NotImplementedError

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _prepare_interactions(self, masks: tf.Tensor, outputs: tf.Tensor) -> Any:
        """Build per-input state shared by the main effects and every pair chunk."""
        return masks, outputs

    def _main_effects(self, state: Any) -> tf.Tensor:
        """Singleton scores aligned with the active channels, shape (d,), float32."""
        return self._estimate(*state)

    def _estimate_pair_chunk(self, state: Any, local_pairs: tf.Tensor) -> tf.Tensor:
        """Pair scores for local active-channel pairs (B, 2), shape (B,), float32."""
        raise NotImplementedError
