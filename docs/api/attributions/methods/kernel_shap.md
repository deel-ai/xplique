# Kernel Shap

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) | <sub><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/kernel_shap.py)

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
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**KernelShap**: Going Further](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT)

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
- \- an identity mapping if inputs has shape $(N, W)$ or $(N, T, W)$, we assume here your inputs
are tabular data or time-series data.

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