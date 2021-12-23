# Kernel Shap

By setting appropriately the perturbation function, the similarity kernel and the interpretable
model in the LIME framework we can theoretically obtain the Shapley Values more efficiently.
Therefore, KernelShap is a method based on LIME with specific attributes.

!!!quote
    The exact computation of SHAP values is challenging. However, by combining insights from current
    additive feature attribution methods, we can approximate them. We describe two model-agnostic
    approximation methods, \[...] and another that is novel (Kernel SHAP)

     -- <cite>[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)</cite>[^1]

## Example

```python
from xplique.attributions import KernelShap

# load images, labels and model
# define a custom map_to_interpret_space function
# ...

method = KernelShap(model, map_to_interpret_space=custom_map)
explanations = method.explain(images, labels)
```

The choice of the map function will have a great deal toward the quality of explanation.
By default, the map function use the quickshift segmentation of scikit-images

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**KernelShap**: Going Further](https://colab.research.google.com/drive/1zTzj1_uTQYQs_7kyhqq_WeBEOy66YeQd) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zTzj1_uTQYQs_7kyhqq_WeBEOy66YeQd) </sub>

{{xplique.attributions.kernel_shap.KernelShap}}

[^1]: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

!!!warning
    The computation time might be very long depending on the hyperparameters settings.
    A huge number of perturbed samples and a fine-grained mapping may lead to better
    results but it is long to compute.

## Parameters in-depth

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
def custom_map_to_interpret_space(inputs: tf.tensor) ->
tf.tensor:
    **some grouping techniques**
    return mappings
```

`mappings` **should have the same dimension as input except for channels**.

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