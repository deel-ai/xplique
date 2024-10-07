# Projections

In example-based explainability, one often needs to define a notion of similarity (distance) between samples. However, the original feature space may not be the most suitable space to define this similarity. For instance, in the case of images, two images can be very similar in terms of their pixel values but very different in terms of their semantic content. In addition, computing distances in the original feature space does not take into account the model's whatsoever, questioning the explainability of the method.

To address these issues, one can project the samples into a new space where the distances between samples are more meaningful with respect to the model's decision. Two approaches are commonly used to define this projection space: (1) use a latent space and (2) use a feature weighting scheme.

Consequently, we defined the general `Projection` class that will be used as a base class for all projection methods. This class allows one to use one or both of the aforementioned approaches. Indeed, one can instantiate a `Projection` object with a `space_projection` method, that define a projection from the feature space to a space of interest, and a`get_weights` method, that defines the feature weighting scheme. The `Projection` class will then project a sample with the `space_projection` method and weight the projected sample's features with the `get_weights` method.

In addition, we provide concrete implementations of the `Projection` class: `LatentSpaceProjection`, `AttributionProjection`, and `HadamardProjection`.

{{xplique.example_based.projections.Projection}}

!!!info
    The `__call__` method is an alias for the `project` method.

## Defining a custom projection

To define a custom projection, one needs to implement the `space_projection` and/or `get_weights` methods. The `space_projection` method should return the projected sample, and the `get_weights` method should return the weights of the features of the projected sample.

!!!info
    The `get_weights` method should take as input the original sample once it has been projected using the `space_projection` method.

For the sake of clarity, we provide an example of a custom projection that projects the samples into a latent space (the final convolution block of the ResNet50 model) and weights the features with the gradients of the model's output with respect to the inputs once they have gone through the layers until the final convolutional layer.

```python
import tensorflow as tf
from xplique.attributions import Saliency
from xplique.example_based.projections import Projection

# load the model
model = tf.keras.applications.ResNet50(weights="imagenet", include_top=True)

latent_layer = model.get_layer("conv5_block3_out") # output of the final convolutional block
features_extractor = tf.keras.Model(
    model.input, latent_layer.output, name="features_extractor"
)

# reconstruct the second part of the InceptionV3 model
second_input = tf.keras.Input(shape=latent_layer.output.shape[1:])

x = second_input
layer_found = False
for layer in model.layers:
    if layer_found:
        x = layer(x)
    if layer == latent_layer:
        layer_found = True

predictor = tf.keras.Model(
    inputs=second_input,
    outputs=x,
    name="predictor"
)

# build the custom projection
space_projection = features_extractor
get_weights = Saliency(predictor)

custom_projection = Projection(space_projection=space_projection, get_weights=get_weights, mappable=False)

# build random samples
rdm_imgs = tf.random.normal((5, 224, 224, 3))
rdm_targets = tf.random.uniform(shape=[5], minval=0, maxval=1000, dtype=tf.int32)
rdm_targets = tf.one_hot(rdm_targets, depth=1000)

# project the samples
projected_samples = custom_projection(rdm_imgs, rdm_targets)
```

{{xplique.example_based.projections.LatentSpaceProjection}}

{{xplique.example_based.projections.AttributionProjection}}

{{xplique.example_based.projections.HadamardProjection}}