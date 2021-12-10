# TCAV

TCAV or Testing with Concept Activation Vector consist consists in using a
concept activation vector (CAV) to quantify the relationship between this
concept and a class.

This is done by using the directional derivative of the concept vector on
several samples of a given class and measuring the percentage of positive
(a positive directional derivative indicating that an infinitesimal addition
of the concept increases the probability of the class).

For a Concept Activation Vector $v_l$ of a layer $f_l$ of a model, and $f_{c}$
the logit of the class $c$, we measure the directional derivative
$S_c(x) = v_l \cdot \frac{ \partial{f_c(x)} } { \partial{f_l}(x) }$.

The TCAV score is the percentage of elements of the class $c$ for which the $S_c$
is positive.

$$ TCAV_c = \frac{|x \in \mathcal{X}^c : S_c(x) > 0 |}{ | \mathcal{X}^c | } $$

## Example

```python
from xplique.concepts import Tcav

tcav_renderer = Tcav(model, 'mixed4d') # you can also pass the layer index (e.g -1)
tcav_score = tcav_renderer(samples, class_index, cav)

```

{{xplique.concepts.tcav.Tcav}}

[^1]: [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) (2018).](https://arxiv.org/abs/1711.11279)
