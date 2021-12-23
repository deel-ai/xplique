# LIME

The Lime method use an interpretable model to provide an explanation.
More specifically, you map inputs ($x \in R^d$) to an interpretable space (e.g super-pixels) of size num_interpetable_features.
From there you generate perturbed interpretable samples ($z' \in \{0,1\}^{num\_interpretable\_samples}$
where $1$ means we keep this specific interpretable feature).

Once you have your interpretable samples you can map them back to their original space
(the perturbed samples $z \in R^d$) and obtain the label prediction of your model for each perturbed
samples.

In the Lime method you define a similarity kernel which compute the similarity between an input and
its perturbed representations (either in the original input space or in the interpretable space):
$\pi_x(z',z)$.

Finally, you train an interpretable model per input, using interpretable samples along the
corresponding perturbed labels and it will draw interpretable samples weighted by the similarity kernel.
Thus, you will have an interpretable explanation (i.e in the interpretable space) which can be
broadcasted afterwards to the original space considering the mapping you used.

!!! quote
    The overall goal of LIME is to identify an interpretable model over the interpretable representation that is locally faithful to the classifier.

     -- <cite>["Why Should I Trust You?": Explaining the Predictions of Any Classifier.](https://arxiv.org/abs/1602.04938)</cite>[^1]

## Example

```python
from xplique.attributions import Lime

# load images, labels and model
# define a custom map_to_interpret_space function
# ...

method = Lime(model, map_to_interpret_space=custom_map)
explanations = method.explain(images, labels)
```

The choice of the interpretable model and the map function will have a great deal toward the quality of explanation.
By default, the map function use the quickshift segmentation of scikit-images

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**LIME**: Going Further](https://colab.research.google.com/drive/1InDzdW39-5k2ENfKqF2bs5qJEv8OJqi2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1InDzdW39-5k2ENfKqF2bs5qJEv8OJqi2) </sub>

{{xplique.attributions.lime.Lime}}

[^1]: ["Why Should I Trust You?": Explaining the Predictions of Any Classifier.](https://arxiv.org/abs/1602.04938)

!!!warning
    The computation time might be very long depending on the hyperparameters settings.
    A huge number of perturbed samples and a fine-grained mapping may lead to better
    results but it is long to compute.

## Parameters in-depth

#### `interpretable_model`:

A Model object providing a `fit` method that train the model with the following inputs:

-   \- `interpretable_inputs`: 2D `ndarray` of shape ($nb\_samples$ x $num\_interp\_features$),
-   \- `expected_outputs`: 1D `ndarray` of shape ($nb\_samples$),
-   \- `weights`: 1D `ndarray` of shape ($nb\_samples$)

The model object should also provide a `predict` and `fit` method.

It should also have a `coef_` attributes (the interpretable explanations) at least
once `fit` is called.

As interpretable model you can use linear models from scikit-learn.

!!!warning
    Note that here `nb_samples` doesn't indicates the length of inputs but the number of
    perturbed samples we want to generate for each input.

#### `similarity_kernel`:

Function which considering an input, perturbed instances of these input and
the interpretable version of those perturbed samples compute the similarities between
the input and the perturbed samples.

!!!info
    The similarities can be computed in the original input space or in the interpretable
    space.

You can provide a custom function. Note that to use a custom function, you have to
follow the following scheme:

```python
def custom_similarity(
    original_input, interpret_samples , perturbed_samples
) -> tf.tensor (shape=(nb_samples,), dtype = tf.float32):
    ** some tf actions **
    return similarities
```

where:

-   \- `original_input` has shape among $(W)$, $(W, H)$, $(W, H, C)$
-   \- `interpret_samples` is a `tf.tensor` of shape $(nb\_samples, num\_interp\_features)$
-   \- `perturbed_samples` is a `tf.tensor` of shape $(nb\_samples, *original\_input.shape)$

If it is possible you can add the `@tf.function` decorator.

!!!warning
    Note that here `nb_samples` doesn't indicates the length of inputs but the number of
    perturbed samples we want to generate for each input.

!!!info
    The default similarity kernel use the euclidean distance between the original input and
    the perturbed samples in the input space.

#### `pertub_func`:

Function which generate perturbed interpretable samples in the interpretation space from
the number of interpretable features (e.g nb of super pixel) and the number of perturbed
samples you want per original input.

The generated `interp_samples` belong to $\{0,1\}^{num\_features}$. Where $1$ indicates that we
keep the corresponding feature (e.g super pixel) in the mapping.

To use your own custom pertub function you should use the following scheme:

```python
@tf.function
def custom_pertub_function(num_features, nb_samples) ->
tf.tensor (shape=(nb_samples, num_interp_features), dtype=tf.int32):
    ** some tf actions**
    return perturbed_sample
```

!!!info
    The default pertub function provided keep a feature (e.g super pixel) with a
    probability 0.5.
    If you want to change it, define the `prob` value when initiating the explainer or define your own function.

#### `map_to_interpret_space`:

Function which group features of an input corresponding to the same interpretable
feature (e.g super-pixel).

It allows to transpose from (resp. to) the original input space to (resp. from)
the interpretable space.

The default mappings are:

- \- the quickshift segmentation algorithm for inputs with $(N, W, H, C)$ shape,
we assume here such shape is used to represent $(W, H, C)$ images.
- \- the felzenszwalb segmentation algorithm for inputs with $(N, W, H)$ shape,
we assume here such shape is used to represent $(W, H)$ images.
- \- an identity mapping if inputs has shape $(N, W)$, we assume here your inputs
are tabular data.

To use your own custom map function you should use the following scheme:

```python
def custom_map_to_interpret_space(single_inp: tf.tensor) ->
tf.tensor:
    **some grouping techniques**
    return mapping
```

`mapping` **should have the same dimension as single input except for channels**.

For instance you can use the scikit-image (as we did for the quickshift algorithm)
library to defines super pixels on your images.

!!!info
    The quality of your explanation relies strongly on this mapping.

!!!warning
    Depending on the mapping you might have a huge number of `interpretable_features` 
    (e.g you map pixels 2 by 2 on a 299x299 image). Thus, the compuation time might
    be very long!

!!!danger
    As you may have noticed, by default **Time Series** are not handled. Consequently, a custom mapping should be implented. Either to assign each feature to a different group or to group consecutive features together, by group of 4 timesteps for example. In the second example, we try to cover patterns. An example is provided below.

```python
def map_time_series(single_input: tf.tensor) -> tf.Tensor:
    time_dim = single_input.shape[0]
    feat_dim = single_input.shape[1]
    mapping = tf.range(time_dim*feat_dim)
    mapping = tf.reshape(mapping, (time_dim, feat_dim))
    return mapping
```