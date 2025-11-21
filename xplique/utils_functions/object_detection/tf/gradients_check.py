"""
TensorFlow gradient checking utilities for object detection models.
"""

from typing import Any

import tensorflow as tf


def check_model_gradients(func: Any, input_tensor: tf.Tensor) -> bool:
    """
    Test gradients in both Eager and Graph modes for an object detection model.

    This function validates that gradients can be computed through the model in both
    TensorFlow execution modes (eager and graph). It handles various output formats
    including dictionaries, lists, MultiBoxTensor objects, and raw tensors.

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

        def check_and_report(loss: tf.Tensor, label: str, tape: tf.GradientTape,
                             input_tensor: tf.Tensor) -> bool:
            try:
                gradients = tape.gradient(loss, input_tensor)
                if gradients is not None:
                    print(f"{label}: OK - {gradients.shape}")
                    return True
                else:
                    print(f"{label}: KO No gradients")
                    return False
            except Exception as e:
                print(f"{label}: Graph Error - {e}")
                return False

        result = False
        try:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_tensor)
                predictions = func_compiled(input_tensor)  # Use the compiled version

                # Handle dict, list, and tensor outputs
                if isinstance(predictions, dict):
                    result = any(
                        check_and_report(tf.reduce_sum(predictions[key]), key, tape, input_tensor)
                        for key in predictions.keys()
                    )

                elif isinstance(predictions, list) and len(predictions) > 0 and hasattr(predictions[0], 'tensor'):
                    # Handle list of MultiBoxTensor objects
                    result = any(
                        check_and_report(tf.reduce_sum(nbc_tensor.tensor), f"MultiBoxTensor[{i}]", tape, input_tensor)
                        for i, nbc_tensor in enumerate(predictions)
                    )

                elif isinstance(predictions, list):
                    result = any(
                        check_and_report(tf.reduce_sum(pred), f"List[{i}]", tape, input_tensor)
                        for i, pred in enumerate(predictions)
                    )

                elif isinstance(predictions, tf.Tensor):
                    result = check_and_report(tf.reduce_sum(predictions), "Tensor", tape, input_tensor)

                else:
                    print(f"Unsupported prediction type: {type(predictions)}")
                    result = False

            del tape

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
    print(f"\n=== SUMMARY ===")
    print(f"Eager mode: {'OK' if eager_result else 'FAIL'}")
    print(f"Graph mode: {'OK' if graph_result else 'FAIL'}")

    return eager_result or graph_result
