# GradCAM

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/grad_cam.py) |
📰 [Paper](https://arxiv.org/abs/1610.02391)

Grad-CAM is a technique for producing visual explanations that can be used on Convolutional Neural
Network (CNN) which uses both gradients and the feature maps of the last conv layer.

!!! quote
    Grad-CAM uses the gradients of any target concept (say logits for “dog” or even a caption), flowing
    into the final convolutional layer to produce a coarse localization map highlighting the important
    regions in the image for predicting the concept.

    -- <cite>[Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)</cite>

More precisely, to obtain the localization map for a prediction $f(x)$, we need to compute the weights
$w_k$ associated to each of the feature map channel $A^k \in \mathbb{R}^{W \times H}$. As we use the last
convolutionnal layer, $k$ will be the number of filters, $Z$ is the number of pixels in each feature
map ($Z = W \times H$, e.g. 7x7 for ResNet50).

$$
w_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial f(x)}{\partial A^k_{i,j}}
$$

We now use this weight to ponderate and aggregate the feature maps to obtain our grad-cam attribution $\phi$:

$$
\phi = \text{max}(0, \sum_k w_k A^k)
$$

Notice that $\phi \in \mathbb{R}^{W \times H}$ and thus the size of the explanation depends on the
size of the feature map ($W, H$) of the last feature map. In order to compare it to the original input $x$,
we upsample $\phi$ using bicubic interpolation.

## Example

```python
from xplique.attributions import GradCAM

# load images, labels and model
# ...

method = GradCAM(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**GradCAM**: Going Further](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X)


{{xplique.attributions.grad_cam.GradCAM}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)
