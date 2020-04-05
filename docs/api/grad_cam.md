# Grad-CAM

Grad-CAM is a technique for producing visual explanations that can be used on Convolutional Neural
Netowrk (CNN) which uses both gradients and the feature maps of the last convolutional layer. 

> Grad-CAM uses the gradients of any target concept (say logits for “dog” or even a caption), flowing 
> into the final convolutional layer to produce a coarse localization map highlighting the important 
> regions in the image for predicting the concept.
>
> -- <cite>[Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)</cite>

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

## Grad-CAM Variations

In the appendix of the original paper [^1] the authors experiments differents procedure for the 
backward pass of ReLU. These two modifications are named Guided-ReLU and Deconv-ReLU.

* **Guided-ReLU** are inspired from Springenberg et al., where to compute the weight $\alpha_k^c$ we 
  only take the positive gradients of the positive activations 
  ($\frac{ \partial{y^c} } { \partial{A_{ij}}} > 0$ and $A_{ij}^k > 0$).
 
* **Deconv-ReLU** are inspired from Zeiler and Fergus, where to compute the weight $\alpha_k^c$ we 
  only take the positive gradients ($\frac{ \partial{y^c} } { \partial{A_{ij}}} > 0$).
  
!!! tip
    As specified in the original paper, the best results are generally obtained by 
    using the Guided-ReLU (see Appendix, B.2 table 3[^1]). That's why the Grad-CAM Api will use this variant
    by default.

## Examples

Using Grad-CAM default variants, Guided ReLU

```python
from xplique.methods import GradCAM

# load images, labels and model
# ...

method = GradCAM(model)
explanations = method.explain(images, labels)
```

Using Grad-CAM procedure as described in the original paper
```python
from Xplique import GradCAM, GradCAMVariants

# load images, labels and model
# ...

method = GradCAM(model, variant=GradCAMVariants.BACKWD_RELU)
explanations = method.explain(images, labels)
```

{{xplique.methods.grad_cam.GradCAM}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)