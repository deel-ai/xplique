# Hsic Attribution Method

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) | 
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/global_sensitivity_analysis/hsic_attribution_method.py) |
📰 [Paper](https://arxiv.org/abs/2206.06219)

The Hsic attribution method from Novello, Fel, Vigouroux[^1] explains a neural network's prediction for a given input image by assessing the dependence between the output and patches of the input. Thanks to the sample efficiency of HSIC Estimator, this black box method requires fewer forward passes to produce relevant explanations.

Let's consider two random variables which are the perturbation associated with each patch of the input image, $X_i, i \in \{1,...d\}$ with $d= \text{grid_size}^2$ image patches and the output $Y$. Let $X^1_i,...,X^p_i$ and $Y^1,...,Y^p$ be $p$ samples of $X_i$ and $Y$. HSIC attribution method requires selecting a kernel for the input and the output to construct an RKHS on which is computed the Maximum Mean Discrepancy, a dissimilarity metric between distributions. Let $k:\mathbb{R}^2 \rightarrow \mathbb{R}$ and $l:\mathbb{R}^2 \rightarrow \mathbb{R}$ the kernels selected for $X_i$ and $Y$, HSIC is estimated with an error $\mathcal{O}(1/\sqrt{p})$ using the estimator 
$$
\mathcal{H}^p_{X_i, Y} = \frac{1}{(p-1)^2} \operatorname{tr} (KHLH),
$$
where $H, L, K \in \mathbb{R}^{p \times p}$ and $K_{ij} = k(x_i, x_j), L_{i,j} = l(y_i, y_j)$ and $H_{ij} = \delta(i=j) - p^{-1}$ where $\delta(i=j) = 1$ if $i=j$ and $0$ otherwise.

In the paper [Making Sense of Dependence: Efficient Black-box Explanations Using Dependence Measure](https://arxiv.org/abs/2206.06219),  the sampler `LatinHypercube` is used to sample the perturbations. Note however that the present implementation uses `TFSobolSequence` as default sampler because `LatinHypercube` requires scipy $\geq$ `1.7.0`. you can nevertheless use this sampler -- which is included in the library -- by specifying it during the init of your explainer. 

For the kernel $k$ applied on $X_i$, a modified Dirac kernel is used to enable an ANOVA-like decomposition property that allows assessing pairwise patch interactions (see the paper for more details). For the kernel $l$ of output $Y$, a Radial Basis Function (RBF) is used.


!!!tip
    We recommend using a grid size of $7 \times 7$ to define the image patches. The paper uses a number of forwards of $1500$ to obtain the most faithful explanations and $750$ for a more budget - but still faithful - version.

!!!info
    To explain small objects in images, it may be necessary to increase the `grid_size`, which also requires an increase in `nb_design`. However, increasing both may impact the memory usage and result in out of memory errors, hence, setting `estimator_batch_size` parameter enables a limited usage of the memory. Note that the classical `batch_size` correspond to the batch_size used in the model call, here `estimator_batch_size` is intern to the method estimator.


## Example

Low budget version

```python
from xplique.attributions import HsicAttributionMethod

# load images, labels and model
# ...

explainer = HsicAttributionMethod(model, grid_size=7, nb_design=750)
explanations = explainer(images, labels)
```

High budget version

```python
from xplique.attributions import HsicAttributionMethod

# load images, labels and model
# ...

explainer = HsicAttributionMethod(model, grid_size=7, nb_design=1500)
explanations = explainer(images, labels)
```

Recommended version, (you need scipy $\geq$ `1.7.0`)

```python
from xplique.attributions import HsicAttributionMethod
from xplique.attributions.global_sensitivity_analysis import LatinHypercube

# load images, labels and model
# ...

explainer = HsicAttributionMethod(model, 
                                  grid_size=7, nb_design=1500,
                                  sampler = LatinHypercube(binary=True))
explanations = explainer(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>


{{xplique.attributions.global_sensitivity_analysis.hsic_attribution_method.HsicAttributionMethod}}

[^1]:[Making Sense of Dependence: Efficient Black-box Explanations Using Dependence Measure (2022)](https://arxiv.org/abs/2206.06219)
