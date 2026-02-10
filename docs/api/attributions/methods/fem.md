# FEM (Feature Explanation Method)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/fem.py) |
📰 [Paper](https://doi.org/10.1109/IPTA50016.2020.9286629)

FEM is a fast, class-agnostic explanation that relies only on activations of a convolutional
layer. For each channel, FEM applies a *k*-sigma threshold to detect unusually strong responses,
builds a binary mask, and aggregates those masks weighted by the channel mean.

More formally, let $A^k \in \mathbb{R}^{H \times W}$ be the $k$-th activation map and define
its mean $\mu_k$ and standard deviation $\sigma_k$ over the spatial dimensions. The threshold is
$T_k = \mu_k + k\sigma_k$ (with $k$ typically 2). The binary mask is $M_k = \mathbb{1}(A^k \geq T_k)$
and the channel weight is $w_k = \mu_k$. The final FEM attribution is

$$
\phi = \sum_k w_k \cdot M_k.
$$

Because no gradients are required, FEM is inexpensive and suitable for inspecting general regions
of interest, though it is class-agnostic by design.

## Limitations / Compatibility

FEM is class-agnostic and does not use labels or targets, so it cannot isolate a specific class.
In Xplique, FEM is implemented for TensorFlow/Keras and expects convolutional feature maps from
image inputs (rank 4 tensors). Models without a convolutional layer are not supported, and FEM
will raise an error if no suitable `conv_layer` can be found.

## Example

```python
from xplique.attributions import FEM

# load images and model
# ...

explainer = FEM(model, k=2.0)
explanations = explainer.explain(images)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)

## References

- Fuad et al., "Features Understanding in 3D CNNs for Actions Recognition in Video" (IPTA 2020, IEEE). [DOI](https://doi.org/10.1109/IPTA50016.2020.9286629)
- Bourroux et al., "Multi Layered Feature Explanation Method for Convolutional Neural Networks" (ICPRAI 2022, LNCS). [DOI](https://doi.org/10.1007/978-3-031-09037-0_49)
- Zhukov et al., "Evaluation of Explanation Methods of AI - CNNs in Image Classification Tasks with Reference-based and No-reference Metrics" (Advances in Artificial Intelligence and Machine Learning, 2023). [DOI](https://doi.org/10.54364/AAIML.2023.1143)
- Zhukov & Benois-Pineau et al., "FEM and Multi-Layered FEM: Feature Explanation Methods with Statistical Filtering of Important Features" in *Emerging Topics in Pattern Recognition and Artificial Intelligence* (World Scientific, 2024). [Book](https://books.google.com/books?id=KQ82EQAAQBAJ)
- Ayyar et al., "Review of white box methods for explanations of convolutional neural networks in image classification tasks" (Journal of Electronic Imaging, 2021). [DOI](https://doi.org/10.1117/1.JEI.30.5.050901)
- Ayyar et al., "A Feature Understanding Method for Explanation of Image Classification by Convolutional Neural Networks" in *Explainable Deep Learning AI* (Academic Press, 2023). [Book](https://www.elsevier.com/educate/ai-and-machine-learning/book/9780323907001/explainable-deep-learning-ai)

{{xplique.attributions.fem.FEM}}
