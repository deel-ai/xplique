# Feature Visualization

One of the specificities of neural networks is their differentiability. This characteristic allows us to compute gradients, either the gradient of a loss with respect to the parameters, or in the case we are interested in here, of a part of the network with respect to the input.
This gradient then allows us to iteratively modify the input in order to maximize an objective such as a neuron, a channel or a combination of objectives.

!!! quote
    If we want to understand individual features, we can search for examples where they have high values
    either for a neuron at an individual position, or for an entire channel.
    -- <cite>[Feature Visualization -- How neural networks build up their understanding of images (2017)](https://distill.pub/2017/feature-visualization)</cite>[^1]

More precisely, the explanation of a neuron $n$ denoted as $\phi^{(n)}$ is an input $x* \in \mathcal{X}$ such that

$$ \phi^{(n)} = \underset{x}{arg\ max}\ f(x)^{(n)} - \mathcal{R}(x) $$

with $f(x)^{(n)}$ the neuron score for an input $x$ and $\mathcal{R}(x)$ a regularization term.
In practice it turns out that preconditioning the input in a decorrelated space such as the frequency domain allows to obtain more consistent results and to better formulate the regularization (e.g. by controlling the rate of high frequency and low frequency desired).

## Examples

Optimize the ten logits of a neural network (we recommend to remove the softmax activation of your network).

```python
from xplique.features_visualizations import Objective
from xplique.features_visualizations import optimize

# load a model...

# targeting the 10 logits of the layer 'logits'
# we can also target a layer by its index, like -1 for the last layer
logits_obj = Objective.neuron(model, "logits", list(range(10)))
images, obj_names = optimize(logits_obj) # 10 images, one for each logits
```

Create a combination of multiple objectives and aggregate them

```python
from xplique.features_visualizations import Objective
from xplique.features_visualizations import optimize

# load a model...

# target the first logits neuron
logits_obj = Objective.neuron(model, "logits", 0)
# target the third layer
layer_obj = Objective.layer(model, "conv2d_1")
# target the second channel of another layer
channel_obj = Objective.channel(model, "mixed4_2", 2)

# combine the objective
obj = logits_obj * 1.0 + layer_obj * 3.0 + channel_obj * (-5.0)
images, obj_names = optimize(logits_obj) # 1 resulting image
```

{{xplique.features_visualizations.objectives.Objective}}

[^1]:[Feature Visualization -- How neural networks build up their understanding of images (2017)](https://distill.pub/2017/feature-visualization)
