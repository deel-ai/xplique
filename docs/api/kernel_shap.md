# Kernel Shap

By setting appropriately the pertubation function, the similarity kernel and the interpretable
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

{{xplique.attributions.kernel_shap.KernelShap}}


[^1]: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
   