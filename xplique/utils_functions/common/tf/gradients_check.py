"""
TensorFlow gradient checking utilities for object detection models.
"""

from typing import Any, List

import tensorflow as tf


def _extract_all_tensors(obj: Any) -> List[tf.Tensor]:
    """
    Recursively extract all TensorFlow tensors from a nested structure.

    Handles nested combinations of dicts, lists, tuples, and tensors.
    Special handling for MultiBoxTensor objects.

    Parameters
    ----------
    obj
        Object to extract tensors from (can be tensor, dict, list, tuple, or nested combinations).

    Returns
    -------
    tensors
        List of all tensors found in the structure.
    """
    if isinstance(obj, tf.Tensor):
        return [obj]
    if hasattr(obj, "tensor"):
        # Handle MultiBoxTensor or similar objects with a .tensor attribute
        return [obj.tensor]
    if isinstance(obj, dict):
        tensors = []
        for value in obj.values():
            tensors.extend(_extract_all_tensors(value))
        return tensors
    if isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(_extract_all_tensors(item))
        return tensors
    # Not a tensor or container, return empty list
    return []


def check_model_gradients(func: Any, input_tensor: tf.Tensor, verbose: bool = False) -> bool:
    """
    Test gradients in both Eager and Graph modes for an object detection model.

    This function validates that gradients can be computed through the model in both
    TensorFlow execution modes (eager and graph). It handles various output formats
    including dictionaries, lists, MultiBoxTensor objects, and raw tensors using
    recursive tensor extraction.

    Parameters
    ----------
    func
        Callable model or function to test. Should accept input_tensor and return
        predictions in dict, list, or tensor format.
    input_tensor
        Input tensor to use for gradient computation testing.
    verbose
        If True, print information about gradient computation. Default is False.

    Returns
    -------
    success
        True if gradients can be computed successfully in at least one mode (eager or graph),
        False otherwise.
    """

    def _test_gradients_single_mode(mode_name: str, func_to_call):
        """
        Test gradients using the provided callable.

        Parameters
        ----------
        mode_name
            Name of the mode for logging ("Eager" or "Graph").
        func_to_call
            The callable to invoke — either the original function (eager)
            or a tf.function-wrapped version (graph).

        Returns
        -------
        success
            True if at least one gradient computation succeeded, False otherwise.
        """
        if verbose:
            print(f"\n--- Testing {mode_name} mode ---")

        result = False
        try:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_tensor)
                predictions = func_to_call(input_tensor)

                # Extract all tensors recursively from the output structure
                tensors = _extract_all_tensors(predictions)

                if not tensors:
                    if verbose:
                        print("No tensors found in outputs")
                    result = False
                else:
                    # Calculate the loss by summing all tensors
                    loss = tf.add_n([tf.reduce_sum(t) for t in tensors])

                    # Compute gradients
                    # pylint: disable=broad-exception-caught
                    try:
                        gradients = tape.gradient(loss, input_tensor)
                        if gradients is not None:
                            if verbose:
                                grad_sum = tf.reduce_sum(tf.abs(gradients))
                                print(f"Gradients OK - sum={grad_sum.numpy():.6f}")
                            result = True
                        else:
                            if verbose:
                                print("No gradients or None gradients")
                            result = False
                    except Exception as e:
                        if verbose:
                            print(f"Gradient computation error: {e}")
                        result = False

            del tape

        # pylint: disable=broad-exception-caught
        except Exception as e:
            if verbose:
                print(f"{mode_name} mode failed completely: {e}")
            result = False

        return result

    # Test both modes — no global state is mutated.
    # Eager: call func directly (TF default behaviour).
    # Graph: tf.function forces graph compilation regardless of any global flag.
    eager_result = _test_gradients_single_mode("Eager", func)
    graph_result = _test_gradients_single_mode("Graph", tf.function(func))

    # Summary
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Eager mode: {'OK' if eager_result else 'FAIL'}")
        print(f"Graph mode: {'OK' if graph_result else 'FAIL'}")

    return eager_result or graph_result
