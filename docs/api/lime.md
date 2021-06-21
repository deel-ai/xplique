# LIME

The Lime method use an interpretable model to provide an explanation.
More specifically, you map inputs ($$x \in R^d$$) to an interpretable space
(e.g super-pixels) of size num_interpetable_features. 
From there you generate pertubed interpretable samples ($$z' \in {0,1}^{num\_interpretable\_samples$$
where $1$ means we keep this specific interpretable feature).
Once you have your interpretable samples you can map them back to their original space
(the pertubed samples $$z \in R^d$$) and obtain the label prediction of your model for each pertubed
samples.
In the Lime method you define a similarity kernel which compute the similarity between an input and
its pertubed representations (either in the original input space or in the interpretable space):
$$\pi_x(z',z)$$.
Finally, you train an interpretable model per input, using interpretable samples along the
corresponding pertubed labels and it will draw interpretable samples weighted by the similarity kernel.
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

method = Lime(model, map_to_interpret_space=custom_map,
    nb_samples=100)
explanations = method.explain(images, labels)
```

The choice of the interpretable model and the map function will have a great deal toward the quality of explanation.

{{xplique.attributions.lime.Lime}}


[^1]: ["Why Should I Trust You?": Explaining the Predictions of Any Classifier.](https://arxiv.org/abs/1602.04938)