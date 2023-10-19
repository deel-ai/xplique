# Guided Backpropagation

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/guided_backpropagation.py) |
📰 [Paper](https://arxiv.org/abs/1412.6806)

Guided-backprop is one of the first attribution method and was proposed in 2014.
Its operation is similar to Saliency: it consists in backpropagating the output score with respect to the input, 
however, at each non-linearity (the ReLUs), only the positive gradient of positive activations are backpropagated.
We can see this as a filter on the backprop.

More precisely, with $f$ our classifier and $f_l(x)$ the activation at layer $l$, we usually have:

$$ \frac{\partial f(x)}{\partial f_{l}(x)} =  \frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \frac{\partial \text{ReLU}(f_l(x))}{\partial f_{l}(x)}
= \frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \odot \mathbb{1}(f_{l}(x))
$$

with $\mathbb{1}(.)$ the indicator function. With Guided-backprop, the backpropagation is modified such that : 

$$
\frac{\partial f(x)}{\partial f_{l}(x)} =  
\frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \odot \mathbb{1}(f_{l}(x)) \odot \mathbb{1}(\frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))})
$$

## Example


```python
from xplique.attributions import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**Guided Backprop**: Going Further](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7)

{{xplique.attributions.gradient_override.guided_backpropagation.GuidedBackprop}}
