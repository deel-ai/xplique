# CAV

CAV or Concept Activation Vector represent a high-level concept as a vector that
indicate the direction to take (for activations of a layer) to maximise this concept.

!!! quote
    \[...] CAV for a concept is simply a vector in the direction of the values
    (e.g., activations) of that concept’s set of examples… we derive CAVs by
    training a linear classifier between a concept’s examples and random counter
    examples and then taking the vector orthogonal to the decision boundary.

     -- <cite>[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) (2018).](https://arxiv.org/abs/1711.11279)</cite>[^1]

For a layer $f_l$ of a model, we seek the linear classifier $v_l \in \mathbb{R}^d$
that separate the activations of the positive examples $\{ f_l(x) : x \in \mathcal{P} \}$,
and the activations of the random/negative examples $\{ f_l(x) : x \in \mathcal{R} \}$.

## Example

```python
from xplique.concepts import Cav

cav_renderer = Cav(model, 'mixed4d', classifier='SGD', test_fraction=0.1)
cav = cav_renderer(positive_examples, random_examples)

```

{{xplique.concepts.cav.Cav}}

[^1]: [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) (2018).](https://arxiv.org/abs/1711.11279)
