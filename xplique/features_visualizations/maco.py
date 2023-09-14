"""
Optimisation functions for MaCo method.
"""

import tensorflow as tf
import numpy as np

from ..types import Optional, Union, Callable, Tuple
from .preconditioning import maco_image_parametrization, init_maco_buffer
from .objectives import Objective


def maco(objective: Objective,
         optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
         nb_steps: int = 256,
         noise_intensity: Optional[Union[float, Callable]] = 0.08,
         box_size: Optional[Union[float, Callable]] = None,
         nb_crops: Optional[int] = 32,
         values_range: Tuple[float, float] = (-1, 1),
         custom_shape: Optional[Tuple] = (512, 512)) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Optimise a single objective using MaCo method. Note that, unlike classic fourier optimization,
    we can only optimize for one objective at a time.

    Ref. Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained
         Optimization (2023).
    https://arxiv.org/abs/2306.06805

    Parameters
    ----------
    objective
        Objective object.
    optimizer
        Optimizer used for gradient ascent, default Nadam(lr=1.0).
    nb_steps
        Number of iterations.
    noise_intensity
        Control the noise injected at each step. Either a float : each step we add noise
        with same std, or a function that associate for each step a noise intensity.
    box_size
        Control the average size of the crop at each step. Either a fixed float (e.g 0.5 means
        the crops will be 50% of the image size) or a function that take as parameter the step
        and return the average box size. Default to linear decay from 50% to 5%.
    nb_crops
        Number of crops used at each steps, higher make the optimisation slower but
        make the results more stable. Default to 32.
    values_range
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).
    custom_shape
        If specified, optimizes images of the given size. Used with
        a low box size to optimize bigger images crop by crop.

    Returns
    -------
    image_optimized
        Optimized image for the given objective.
    transparency
        Transparency of the image, i.e the sum of the absolute value of the gradients
        of the image with respect to the objective.
    """
    values_range = (min(values_range), max(values_range))

    model, objective_function, _, input_shape = objective.compile()

    assert input_shape[0] == 1, "You can only optimize one objective at a time with MaCo."

    if optimizer is None:
        optimizer = tf.keras.optimizers.Nadam(1.0)

    if box_size is None:
        # default to box_size that go from 50% to 5%
        box_size_values = tf.cast(np.linspace(0.5, 0.05, nb_steps), tf.float32)
        get_box_size = lambda step_i: box_size_values[step_i]
    elif hasattr(box_size, "__call__"):
        get_box_size = box_size
    elif isinstance(box_size, float):
        get_box_size = lambda _ : box_size
    else:
        raise ValueError('box_size must be a function or a float.')

    if noise_intensity is None:
        # default to large noise to low noise
        noise_values = tf.cast(np.logspace(0, -4, nb_steps), tf.float32)
        get_noise_intensity = lambda step_i: noise_values[step_i]
    elif hasattr(noise_intensity, "__call__"):
        get_noise_intensity = noise_intensity
    elif isinstance(noise_intensity, float):
        get_noise_intensity = lambda _ : noise_intensity
    else:
        raise ValueError('noise_intensity size must be a function or a float.')

    img_shape = (input_shape[1], input_shape[2])
    if custom_shape:
        img_shape = custom_shape

    magnitude, phase = init_maco_buffer(img_shape)
    phase = tf.Variable(phase, trainable=True)

    transparency = tf.zeros((*custom_shape, 3))

    for step_i in range(nb_steps):

        box_size_at_i = get_box_size(step_i)
        noise_intensity_at_i = get_noise_intensity(step_i)

        grads, grads_img = maco_optimisation_step(model, objective_function, magnitude, phase,
                                                  box_size_at_i, noise_intensity_at_i, nb_crops,
                                                  values_range)

        optimizer.apply_gradients(zip([-grads], [phase]))
        transparency += tf.abs(grads_img)

    img = maco_image_parametrization(magnitude, phase, values_range).numpy()

    return img, transparency


@tf.function
def maco_optimisation_step(model, objective_function, magnitude, phase,
                           box_average_size, noise_std, nb_crops, values_range):
    """ Optimisation step for MaCo method.

    Parameters
    ----------
    model
        Model to optimize on.
    objective_function
        Objective function (e.g neurons, channel).
    magnitude
        Fixed magnitude used to generate the image (the magnitude is fixed and not optimized).
    phase
        Phase of the image of the current image in fourier domain.
    box_average_size
        Average size of the crops.
    noise_std
        Standard deviation of the noise added at each step.
    nb_crops
        Number of crops to use for each step.
    values_range
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).

    Returns
    -------
    grads
        Gradients of the phase with respect to the objective.
    grads_img
        Gradients of the image with respect to the objective.
    """
    with tf.GradientTape() as tape:
        tape.watch(phase)

        image = maco_image_parametrization(magnitude, phase, values_range)

        # sample random crops in the buffer
        center_x = 0.5 + tf.random.normal((nb_crops,), stddev=0.15)
        center_y = 0.5 + tf.random.normal((nb_crops,), stddev=0.15)
        delta_x = tf.random.normal((nb_crops,), stddev=0.05, mean=box_average_size)
        delta_x = tf.clip_by_value(delta_x, 0.05, 1.0)
        delta_y = delta_x  # square boxes

        box_indices = tf.zeros(shape=(nb_crops,), dtype=tf.int32)
        boxes = tf.stack([center_x - delta_x * 0.5,
                          center_y - delta_y * 0.5,
                          center_x + delta_x * 0.5,
                          center_y + delta_y * 0.5], -1)

        crops = tf.image.crop_and_resize(image[None, :, :, :], boxes, box_indices,
                                         model.input_shape[1:3])

        # add random noise as feature viz robustness
        # see (https://distill.pub/2017/feature-visualization)
        crops += tf.random.normal(crops.shape, stddev=noise_std, mean=0.0)
        crops += tf.random.uniform(crops.shape, minval=-noise_std/2.0,
                                                maxval=noise_std/2.0)

        # maco is always is a single objective
        model_outputs = model(crops)
        loss = tf.reduce_mean(objective_function([model_outputs]))

    # also get the gradient to the image for transparency
    grads = tape.gradient(loss, [phase, image])
    grads_phase, grads_image = grads

    return grads_phase, grads_image
