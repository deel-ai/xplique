"""
Optimisation functions
"""

import tensorflow as tf

from ..commons.model_override import override_relu_gradient, open_relu_policy
from ..types import Optional, Union, List, Callable, Tuple
from .preconditioning import fft_image, get_fft_scale, fft_to_rgb, to_valid_rgb
from .transformations import generate_standard_transformations
from .objectives import Objective


def optimize(objective: Objective,
             optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
             nb_steps: int = 256,
             use_fft: bool = True,
             fft_decay: float = 0.85,
             std: float = 0.01,
             regularizers: Optional[List[Callable]] = None,
             image_normalizer: str = 'sigmoid',
             values_range: Tuple[float, float] = (0, 1),
             transformations: Optional[Union[List[Callable], str]] = 'standard',
             warmup_steps: int = False,
             custom_shape: Optional[Tuple] = (512, 512),
             save_every: Optional[int] = None) -> Tuple[List[tf.Tensor], List[str]]:
             # pylint: disable=R0913,E1130
    """
    Optimise a given objective using gradient ascent.

    Parameters
    ----------
    objective
        Objective object.
    optimizer
        Optimizer used for gradient ascent.
    nb_steps
        Number of iterations.
    use_fft
        If true, use fourier preconditioning.
    fft_decay
        Control the allowed energy of the high frequency, a high value
        suppresses high frequencies.
    std
        Standard deviation used for the image initialization (or buffer for fft).
    regularizers
        List of regularizers that are applied on the image and added to the loss.
    image_normalizer
        Transformation applied to the image after each iterations to ensure the
        pixels are in [0,1].
    values_range
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).
    transformations
        Transformations applied to the image during optimisation, default to robust standard
        transformations.
    warmup_steps
        If true, clone the model by replacing the Relu's with Leaky Relu's to
        find a pre-optimised image, allowing the visualization process to get
        started (as the relu could block the flow of gradient).
    custom_shape
        If specified, optimizes images of the given size. Often use with
        jittering & scale to optimize bigger images crop by crop.
    save_every
        Define the steps to which we save the optimized images. In any case, the
        last images will be returned.

    Returns
    -------
    images_optimized
        Optimized images for each objectives.
    objective_names
        Name of each objectives.
    """
    model, objective_function, objective_names, input_shape = objective.compile()

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(0.05)

    img_shape = input_shape
    if custom_shape:
        img_shape = (img_shape[0], *custom_shape, img_shape[-1])

    if transformations == 'standard':
        transformations = generate_standard_transformations(img_shape[1])

    if use_fft:
        inputs = tf.Variable(fft_image(img_shape, std), trainable=True)
        fft_scale = get_fft_scale(img_shape[1], img_shape[2], decay_power=fft_decay)
        image_param = lambda inputs: to_valid_rgb(fft_to_rgb(img_shape, inputs, fft_scale),
                                                  image_normalizer, values_range)
    else:
        inputs = tf.Variable(tf.random.normal(img_shape, std, dtype=tf.float32))
        image_param = lambda inputs: to_valid_rgb(inputs, image_normalizer, values_range)

    optimisation_step = _get_optimisation_step(objective_function,
                                               len(model.outputs),
                                               image_param,
                                               input_shape,
                                               transformations,
                                               regularizers)

    if warmup_steps:
        model_warmup = override_relu_gradient(model, open_relu_policy)
        for _ in range(warmup_steps):
            grads = optimisation_step(model_warmup, inputs)
            optimizer.apply_gradients([(-grads, inputs)])

    images_optimized = []
    for step_i in range(nb_steps):
        grads = optimisation_step(model, inputs)
        optimizer.apply_gradients([(-grads, inputs)])

        last_iteration = step_i == nb_steps - 1
        should_save = save_every and (step_i + 1) % save_every == 0
        if should_save or last_iteration:
            imgs = image_param(inputs)
            images_optimized.append(imgs)

    return images_optimized, objective_names


def _get_optimisation_step(
        objective_function: Callable,
        nb_outputs: int,
        image_param: Callable,
        input_shape: Tuple,
        transformations: Optional[Callable] = None,
        regularizers: Optional[List[Callable]] = None) -> Callable:
    """
    Generate a function that optimize the objective function for a single step.

    Parameters
    ----------
    objective_function
        Function that compute the loss for the objectives given the model
        outputs.
    nb_outputs
        Number of outputs of the model.
    image_param
        Function that map image to a valid rgb.
    input_shape
        Shape of the inputs to optimize.
    transformations
        Transformations applied to the image during optimisation.
    regularizers
        List of regularizers that are applied on the image and added to the loss.

    Returns
    -------
    step_function
        Function (model, inputs) to call to optimize the input for one step.
    """

    @tf.function
    def step(model, inputs):

        with tf.GradientTape() as tape:
            tape.watch(inputs)

            imgs = image_param(inputs)
            if transformations:
                imgs = transformations(imgs)
            imgs = tf.image.resize(imgs, (input_shape[1], input_shape[2]))

            model_outputs = model(imgs)

            if nb_outputs == 1:
                model_outputs = tf.expand_dims(model_outputs, 0)

            loss = objective_function(model_outputs)

            if regularizers:
                for reg_function in regularizers:
                    loss -= reg_function(imgs)

        grads = tape.gradient(loss, inputs)

        return grads

    return step
