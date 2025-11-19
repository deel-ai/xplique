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
    if hasattr(obj, 'tensor'):
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


def check_model_gradients(func: Any, input_tensor: tf.Tensor) -> bool:
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

    Returns
    -------
    success
        True if gradients can be computed successfully in eager mode, False otherwise.
    """

    def _test_gradients_single_mode(mode_name: str, eager_mode: bool):
        """
        Test gradients in a specific execution mode.

        Parameters
        ----------
        mode_name
            Name of the mode for logging ("Eager" or "Graph").
        eager_mode
            If True, test in eager mode. If False, test in graph mode.

        Returns
        -------
        success
            True if at least one gradient computation succeeded, False otherwise.
        """
        print(f"\n--- Testing {mode_name} mode ---")

        # Set the execution mode
        original_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(eager_mode)

        # Force compilation in Graph mode
        if not eager_mode:
            func_compiled = tf.function(func)  # Force Graph compilation
        else:
            func_compiled = func

        result = False
        try:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_tensor)
                predictions = func_compiled(input_tensor)  # Use the compiled version

                # Extract all tensors recursively from the output structure
                tensors = _extract_all_tensors(predictions)

                if not tensors:
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
                            grad_sum = tf.reduce_sum(tf.abs(gradients))
                            print(f"Gradients OK - sum={grad_sum.numpy():.6f}")
                            result = True
                        else:
                            print("No gradients or None gradients")
                            result = False
                    except Exception as e:
                        print(f"Gradient computation error: {e}")
                        result = False

            del tape

        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"{mode_name} mode failed completely: {e}")
            result = False
        finally:
            # Restore original mode
            tf.config.run_functions_eagerly(original_eager)

        return result

    # Test both modes
    eager_result = _test_gradients_single_mode("Eager", True)
    graph_result = _test_gradients_single_mode("Graph", False)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Eager mode: {'OK' if eager_result else 'FAIL'}")
    print(f"Graph mode: {'OK' if graph_result else 'FAIL'}")

    return eager_result or graph_result
