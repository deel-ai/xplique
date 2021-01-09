"""
Optimisation functions
"""

import numpy as np
import tensorflow as tf

from ..utils.model_override import override_relu_gradient
from .preconditioning import fft_image, get_fft_scale, fft_to_rgb, to_valid_rgb
from .transformations import standard_transformations


def optimize(objective,
             optimizer,
             nb_steps=256,
             use_fft=True,
             fft_decay=1.0,
             std=0.5,
             regularizers=None,
             image_normalizer='sigmoid',
             transformations=standard_transformations,
             warmup_steps=16,
             custom_shape=None,
             save_every=None,
             ): # pylint: disable=R0913
    """
    Optimise a given objective using gradient ascent.

    Parameters
    ----------
    objective : Objective
        Objective object.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer used for gradient ascent.
    nb_steps : int, optional
        Number of iterations.
    use_fft : bool, optional
        If true, use fourier preconditioning.
    fft_decay : float, optional
        Control the allowed energy of the high frequency, a high value
        suppresses high frequencies.
    std : float, optional
        Standard deviation used for the image initialization (or buffer for fft).
    regularizers : list of function, optional
        List of regularizers that are applied on the image and added to the loss.
    image_normalizer : None, 'sigmoid', 'clip', optional
        Transformation applied to the image after each iterations to ensure the
        pixels are in [0,1].
    transformations : function, optional
        Transformations applied to the image during optimisation.
    warmup_steps : bool, optional
        If true, clone the model by replacing the Relu's with Leaky Relu's to
        find a pre-optimised image, allowing the visualization process to get
        started (as the relu could block the flow of gradient).
    custom_shape : tuple (width, height), optional
        If specified, optimizes images of the given size. Often use with
        jittering & scale to optimize bigger images crop by crop.
    save_every : int, optional
        Define the steps to which we save the optimized images. In any case, the
        last images will be returned.

    Returns
    -------
    images_optimized : ndarray (M, N, W, H, C)
        Optimized images, with M the number of saving and N the number of
        objectives.
    objective_names : list of string (N)
        Name of each objectives.
    """
    model, objective_function, objective_names, input_shape = objective.compile()

    img_shape = input_shape
    if custom_shape:
        img_shape = (img_shape[0], *custom_shape, img_shape[-1])

    if use_fft:
        inputs = tf.Variable(fft_image(img_shape, std), trainable=True)
        fft_scale = get_fft_scale(img_shape[1], img_shape[2], decay_power=fft_decay)
        image_param = lambda inputs: to_valid_rgb(fft_to_rgb(img_shape, inputs, fft_scale),
                                                  image_normalizer)
    else:
        inputs = tf.Variable(tf.random.normal(img_shape, std, dtype=tf.float32))
        image_param = lambda inputs: to_valid_rgb(inputs, image_normalizer)

    optimisation_step = _get_optimisation_step(objective_function,
                                               len(model.outputs),
                                               image_param,
                                               input_shape,
                                               transformations,
                                               regularizers)

    if warmup_steps:
        model_warmup = override_relu_gradient(model, tf.nn.leaky_relu)
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

    return np.array(images_optimized), objective_names


def _get_optimisation_step(
        objective_function,
        nb_outputs,
        image_param,
        input_shape,
        transformations=None,
        regularizers=None):
    """
    Generate a function that optimize the objective function for a single step.

    Parameters
    ----------
    objective_function : function
        Function that compute the loss for the objectives given the model
        outputs.
    nb_outputs : int
        Number of outputs of the model.
    image_param : function
        Function that map image to a valid rgb.
    input_shape : tuple (N, W, H, C)
        Shape of the inputs to optimize.
    transformations : function, optional
        Transformations applied to the image during optimisation.
    regularizers : list of function, optional
        List of regularizers that are applied on the image and added to the loss.

    Returns
    -------
    step_function : function
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
