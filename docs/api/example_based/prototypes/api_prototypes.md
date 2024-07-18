# Prototypes
Prototype-based explanation is a family of natural example-based XAI methods. Prototypes consist of a set of samples that are representative of either the dataset or a class. Three classes of prototype-based methods are found in the literature ([Poché et al., 2023](https://hal.science/hal-04117520/document)): [Prototypes for Data-Centric Interpretability](#prototypes-for-data-centric-interpretability), [Prototypes for Post-hoc Interpretability](#prototypes-for-post-hoc-interpretability) and Prototype-Based Models Interpretable by Design. This library focuses on first two classes.

## Prototypes for Data-Centric Interpretability
In this class, prototypes are selected without relying on the model and provide an overview of
the dataset. As mentioned in ([Poché et al., 2023](https://hal.science/hal-04117520/document)), we found `clustering methods`, `set cover methods` and `data summarization methods`. This library focuses on `data summarization methods`, also known as `set cover problem methods`, which can be treated in two ways [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf): 

- **Summarization with knapsack constraint**: 
consists in finding a subset of prototypes $\mathcal{P}$ that maximizes the coverage set function $F(\mathcal{P})$ under the constraint that its selection cost $C(\mathcal{P})$ (e.g., the number of selected prototypes $|\mathcal{P}|$) should be less than a given budget. 

- **Summarization with covering constraint**:
consists in finding a low-cost subset under the constraint it should cover all the data. For both cases, submodularity and monotonicity of $F(\mathcal{P})$ are necessary to guarantee that a greedy algorithm has a constant factor guarantee of optimality [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf).  In addition, $F(\mathcal{P})$ should encourage coverage and penalize redundancy in order to have a good summary [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf).

This library implements three methods from **Summarization with knapsack constraint**: `MMDCritic`, `ProtoGreedy` and `ProtoDash`.
[Kim et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf) proposed `MMDCritic` method that used a set function based on the Maximum Mean Discrepancy [(MMD)](#what-is-mmd). They added additional diagonal dominance conditions on the kernel to ensure monotonocity and submodularity. They solve summarization with knapsack constraint problem to find both prototypes and criticisms. First, the number of prototypes and criticisms to be found, respectively as $m_p$ and $m_c$, are selected. Second, to find prototypes, a greedy algorithm is used to maximize $F(\mathcal{P})$ s.t. $|\mathcal{P}| \le m_p$ where $F(\mathcal{P})$ is defined as:
\begin{equation}
    F(\mathcal{P})=\frac{2}{|\mathcal{P}|\cdot n}\sum_{i,j=1}^{|\mathcal{P}|,n}\kappa(p_i,x_j)-\frac{1}{|\mathcal{P}|^2}\sum_{i,j=1}^{|\mathcal{P}|}\kappa(p_i,p_j)
\end{equation}
Finally, to find criticisms $\mathcal{C}$, the same greedy algorithm is used to select points that maximize another objective function $J(\mathcal{C})$.

[Gurumoorthy et al., 2019](https://arxiv.org/pdf/1707.01212) associated non-negative weights to prototypes which are indicative of their importance. In this way, both prototypes and criticisms (which are the least weighted examples from prototypes) can be found by maximizing the same set function $F(\mathcal{P})$. They established the weak submodular property of $J(\mathcal{P})$ and present tractable algorithms (`ProtoGreedy` and `ProtoDash`) to optimize it. Their method works for any symmetric positive definite kernel which is not the case for `MMDCritic`. First, they define a weighted objective $F(\mathcal{P},w)$:
\begin{equation}   
F(\mathcal{P},w)=\frac{2}{n}\sum_{i,j=1}^{|\mathcal{P}|,n}w_i\kappa(p_i,x_j)-\sum_{i,j=1}^{|\mathcal{P}|}w_iw_j\kappa(p_i,p_j),
\end{equation}
where $w$ are non-negative weights for each prototype. Then, they find $\mathcal{P}$ with a corresponding $w$ that maximizes $J(\mathcal{P}) \equiv \max_{w:supp(w)\in \mathcal{P},w\ge 0} J(\mathcal{P},w)$ s.t. $|\mathcal{P}| \leq m=m_p+m_c$. $J(\mathcal{P})$ can be maximized either by `ProtoGreedy` or by `ProtoDash`. `ProtoGreedy` selects the next element that maximizes the increment of the scoring function while `Protodash` selects the next element that maximizes the gradient of $F(\mathcal{P},w)$ with respect to $w$. `ProtoDash` is much faster than `ProtoGreedy` without compromising on the quality of the solution (the complexity of `ProtoGreedy` is $O(n(n+m^4))$ comparing to $O(n(n+m^2)+m^4)$ for `ProtoDash`). The difference between `ProtoGreedy` and the greedy algorithm of `MMDCritic` is that `ProtoGreedy` additionally determines the weights for each of the selected prototypes. The approximation guarantee is $(1-e^{-\gamma})$ for `ProtoGreedy`, where $\gamma$ is submodularity ratio of $F(\mathcal{P})$, comparing to $(1-e^{-1})$ for `MMDCritic`. 

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

!!!warning
    For `MMDCritic`, the kernel must satisfy a condition ensuring the submodularity of the set function (the Gaussian kernel respects this constraint). In contrast, for `Protodash` and `Protogreedy`, any kernel can be used, as these methods rely on weak submodularity instead of full submodularity.

### Default kernel
The default kernel used is Gaussian kernel. This kernel distance assigns higher similarity to points that are close in feature space and gradually decreases similarity as points move further apart. It is a good choice when your data has complexity. However, it can be sensitive to the choice of hyperparameters, such as the width $\sigma$ of the Gaussian kernel, which may need to be carefully fine-tuned.

The Data-Centric prototypes methods are implemented as [search methods](../../xplique/example_based/search_methods/):

| Method Name and Documentation link     | **Tutorial**             | Available with TF | Available with PyTorch* |
|:-------------------------------------- | :----------------------: | :---------------: | :---------------------: |
| [ProtoGreedySearch](../proto_greedy/)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
| [ProtoDashSearch](../proto_dash/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
| [MMDCriticSearch](../mmd_critic/)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |

*: Before using a PyTorch model it is highly recommended to read the [dedicated documentation](../pytorch/)

The class `ProtoGreedySearch` inherits from the `BaseSearchMethod` class. It finds prototypes and assigns a non-negative weight to each one.

Both the `MMDCriticSearch` and `ProtoDashSearch` classes inherit from the `ProtoGreedySearch` class.

The class `MMDCriticSearch` differs from `ProtoGreedySearch` by assigning equal weights to the selection of prototypes. The two classes use the same greedy algorithm. In the `compute_objective` method of `ProtoGreedySearch`, for each new candidate, we calculate the best weights for the selection of prototypes. However, in `MMDCriticSearch`, the `compute_objective` method assigns the same weight to all elements in the selection.

The class `ProtoDashSearch`, like `ProtoGreedySearch`, assigns a non-negative weight to each prototype. However, the algorithm used by `ProtoDashSearch` is different: it maximizes a tight lower bound on $l(w)$ instead of maximizing $l(w)$, as done in `ProtoGreedySearch`. Therefore, `ProtoDashSearch` overrides the `compute_objective` method to calculate an objective based on the gradient of $l(w)$. It also overrides the `update_selection` method to select the best weights of the selection based on the gradient of the best candidate.

## Prototypes for Post-hoc Interpretability

Data-Centric methods such as `Protogreedy`, `ProtoDash` and `MMDCritic` can be used in either the output or the latent space of the model. In these cases, [projections methods](./algorithms/projections/) are used to transfer the data from the input space to the latent/output spaces.

The search method can have attribute `projection` that projects samples to a space where distances between samples make sense for the model. Then the `search_method` finds the prototypes by looking in the projected space.




