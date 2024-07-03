# Prototypes

Prototype-based explanation is a family of natural example-based XAI methods. Prototypes consist of a set of samples that are representative of either the dataset or a class.

Three classes of prototype-based methods are found in the literature ([Poché et al., 2023](https://hal.science/hal-04117520/document)): Prototypes for Data-Centric Interpretability, Prototypes for Post-hoc Interpretability and Prototype-Based Models Interpretable by Design. This library focuses on first two classes.

## Prototypes for Data-Centric Interpretability
In this class, prototypes are selected without relying on the model and provide an overview of
the dataset. In this library, the following methode are implemented as [search methods](./algorithms/search_methods/):

Xplique includes the following prototypes search methods:

| Method Name and Documentation link     | **Tutorial**             | Available with TF | Available with PyTorch* |
|:-------------------------------------- | :----------------------: | :---------------: | :---------------------: |
| [ProtoGreedySearch](../proto_greedy_search/)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
| [ProtoDashSearch](../proto_dash_search/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
| [MMDCriticSearch](../mmd_critic_search/)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |

*: Before using a PyTorch model it is highly recommended to read the [dedicated documentation](../pytorch/)

### What is MMD?
The commonality among these three methods is their utilization of the Maximum Mean Discrepancy (MMD) statistic as a measure of similarity between points and potential prototypes. MMD is a statistic for comparing two distributions (similar to KL-divergence). However, it is a non-parametric statistic, i.e., it does not assume a specific parametric form for the probability distributions being compared. It is defined as follows:

$$
\begin{align*}
\text{MMD}(P, Q) &= \left\| \mathbb{E}_{X \sim P}[\varphi(X)] - \mathbb{E}_{Y \sim Q}[\varphi(Y)] \right\|_\mathcal{H}
\end{align*}
$$

where $\varphi(\cdot)$ is a mapping function of the data points. If we want to consider all orders of moments of the distributions, the mapping vectors $\varphi(X)$ and $\varphi(Y)$ will be infinite-dimensional. Thus, we cannot calculate them directly. However, if we have a kernel that gives the same result as the inner product of these two mappings in Hilbert space ($k(x, y) = \langle \varphi(x), \varphi(y) \rangle_\mathcal{H}$), then the $MMD^2$ can be computed using only the kernel and without explicitly using $\varphi(X)$ and $\varphi(Y)$ (this is called the kernel trick):

$$
\begin{align*}
\text{MMD}^2(P, Q) &= \langle \mathbb{E}_{X \sim P}[\varphi(X)], \mathbb{E}_{X' \sim P}[\varphi(X')] \rangle_\mathcal{H} + \langle \mathbb{E}_{Y \sim Q}[\varphi(Y)], \mathbb{E}_{Y' \sim Q}[\varphi(Y')] \rangle_\mathcal{H} \\
&\quad - 2\langle \mathbb{E}_{X \sim P}[\varphi(X)], \mathbb{E}_{Y \sim Q}[\varphi(Y)] \rangle_\mathcal{H} \\
&= \mathbb{E}_{X, X' \sim P}[k(X, X')] + \mathbb{E}_{Y, Y' \sim Q}[k(Y, Y')] - 2\mathbb{E}_{X \sim P, Y \sim Q}[k(X, Y)]
\end{align*}
$$

### How to choose the kernel ?
The choice of the kernel for selecting prototypes depends on the specific problem and the characteristics of your data. Several kernels can be used, including:

- Gaussian
- Laplace
- Polynomial
- Linear...

If we consider any exponential kernel (Gaussian kernel, Laplace, ...), we automatically consider all the moments for the distribution, as the Taylor expansion of the exponential considers infinite-order moments. It is better to use a non-linear kernel to capture non-linear relationships in your data. If the problem is linear, it is better to choose a linear kernel such as the dot product kernel, since it is computationally efficient and often requires fewer hyperparameters to tune.

For the MMD-critic method, the kernel must satisfy a condition ensuring the submodularity of the set function (the Gaussian kernel respects this constraint). In contrast, for Protodash and Protogreedy, any kernel can be used, as these methods rely on weak submodularity instead of full submodularity.

### Default kernel
The default kernel used is Gaussian kernel. This kernel distance assigns higher similarity to points that are close in feature space and gradually decreases similarity as points move further apart. It is a good choice when your data has complexity. However, it can be sensitive to the choice of hyperparameters, such as the width $\sigma$ of the Gaussian kernel, which may need to be carefully fine-tuned.

## Prototypes for Post-hoc Interpretability

Data-Centric methods such as Protogreedy, ProtoDash and MMD-critic can be used in either the output or the latent space of the model. In these cases, [projections methods](./algorithms/projections/) are used to transfer the data from the input space to the latent/output spaces.

# Architecture of the code

The Data-Centric prototypes methods are implemented as `search_methods`. The search method can have attribute `projection` that projects samples to a space where distances between samples make sense for the model. Then the `search_method` finds the prototypes by looking in the projected space.

The class `ProtoGreedySearch` inherits from the `BaseSearchMethod` class. It finds prototypes and assigns a non-negative weight to each one.

Both the `MMDCriticSearch` and `ProtoDashSearch` classes inherit from the `ProtoGreedySearch` class.

The class `MMDCriticSearch` differs from `ProtoGreedySearch` by assigning equal weights to the selection of prototypes. The two classes use the same greedy algorithm. In the `compute_objective` method of `ProtoGreedySearch`, for each new candidate, we calculate the best weights for the selection of prototypes. However, in `MMDCriticSearch`, the `compute_objective` method assigns the same weight to all elements in the selection.

The class `ProtoDashSearch`, like `ProtoGreedySearch`, assigns a non-negative weight to each prototype. However, the algorithm used by `ProtoDashSearch` is different: it maximizes a tight lower bound on $l(w)$ instead of maximizing $l(w)$, as done in `ProtoGreedySearch`. Therefore, `ProtoDashSearch` overrides the `compute_objective` method to calculate an objective based on the gradient of $l(w)$. It also overrides the `update_selection` method to select the best weights of the selection based on the gradient of the best candidate.