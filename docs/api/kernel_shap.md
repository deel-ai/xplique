# Kernel Shap

Kernel SHAP is a method that uses the LIME framework to compute Shapley Values. Setting
the pertubation function and the similarity kernel appropriately in the LIME framework
allows theoretically obtaining Shapley Values more efficiently than directly computing
Shapley Values.

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

method = KernelShap(model, map_to_interpret_space=custom_map,
    nb_samples=100)
explanations = method.explain(images, labels)
```

The choice of the map function will have a great deal toward the quality of explanation.

{{xplique.attributions.kernel_shap.KernelShap}}


[^1]: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
   