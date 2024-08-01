# Projections

In example-based explainability, one often needs to define a notion of similarity (distance) between samples. However, the original feature space may not be the most suitable space to define this similarity. For instance, in the case of images, two images can be very similar in terms of their pixel values but very different in terms of their semantic content. In addition, computing distances in the original feature space does not take into account the model's whatsoever, questioning the explainability of the method.

To address these issues, one can project the samples into a new space where the distances between samples are more meaningful with respect to the model's decision. Two approaches are commonly used to define this projection space: (1) use a latent space and (2) use a feature weighting scheme.

Consequently, we defined the general `Projection` class that will be used as a base class for all projection methods. This class allows one to use one or both of the aforementioned approaches. Indeed, one can instantiate a `Projection` object with a `space_projection` method, that define a projection from the feature space to a space of interest, and a`get_weights` method, that defines the feature weighting scheme. The `Projection` class will then project a sample with the `space_projection` method and weight the projected sample's features with the `get_weights` method.

In addition, we provide concrete implementations of the `Projection` class: `LatentSpaceProjection`, `AttributionProjection`, and `HadamardProjection`.

## `Projection` class

{{xplique.example_based.projections.Projection}}

## `LatentSpaceProjection` class

{{xplique.example_based.projections.LatentSpaceProjection}}

## `AttributionProjection` class

{{xplique.example_based.projections.AttributionProjection}}

## `HadamardProjection` class

{{xplique.example_based.projections.HadamardProjection}}
