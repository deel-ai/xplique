import tensorflow as tf

def check_model_gradients(func, input_tensor: tf.Tensor) -> bool:
    """Test gradients in both Eager and Graph modes"""

    def _test_gradients_single_mode(mode_name: str, eager_mode: bool):
        """Test gradients in a specific mode"""
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

                # Handle both dict and tensor outputs
                if isinstance(predictions, dict):
                    for key in predictions.keys():
                        try:
                            loss = tf.reduce_sum(predictions[key])
                            gradients = tape.gradient(loss, input_tensor)

                            if gradients is not None:
                                print(f"{key}: OK - {gradients.shape}")
                                result = True
                            else:
                                print(f"{key}: KO No gradients")
                        except Exception as e:
                            print(f"{key}: Graph Error - {e}")

                elif isinstance(predictions, tf.Tensor):
                    try:
                        loss = tf.reduce_sum(predictions)
                        gradients = tape.gradient(loss, input_tensor)

                        if gradients is not None:
                            print(f"Tensor: OK - {gradients.shape}")
                            result = True
                        else:
                            print(f"Tensor: KO No gradients")
                    except Exception as e:
                        print(f"Tensor: Graph Error - {e}")
                else:
                    print(f"Unsupported prediction type: {type(predictions)}")

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
    
    return eager_result and graph_result