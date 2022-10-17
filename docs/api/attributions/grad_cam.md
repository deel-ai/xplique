# GradCAM

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/grad_cam.py)

Grad-CAM is a technique for producing visual explanations that can be used on Convolutional Neural
Netowrk (CNN) which uses both gradients and the feature maps of the last convolutional layer.

!!! quote
    Grad-CAM uses the gradients of any target concept (say logits for “dog” or even a caption), flowing
    into the final convolutional layer to produce a coarse localization map highlighting the important
    regions in the image for predicting the concept.

    -- <cite>[Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)</cite>

More precisely, to obtain the localization map for a class $L_c$, we need to compute the weights
$\alpha_k^c$ associated to each of the feature map activation $A^k$. As we use the last
convolutionnal layer, $k$ will be the number of filters, $Z$ is the number of pixels in each feature
map (width $\cdot$ height).

\begin{align}
 \alpha_k^c = \frac{1}{Z} \sum_i\sum_j \frac{ \partial{y^c}} {\partial{A_{ij}^k} } \\
 L^c = max(0, \sum_k \alpha_k^c A^k)
\end{align}

Notice that the size of the explanation depends on the size (height, width) of the last feature map,
so we have to interpolate in order to find the same dimensions as the input.

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
- [**GradCAM**: Going Further](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X?authuser=1)


{{xplique.attributions.grad_cam.GradCAM}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)