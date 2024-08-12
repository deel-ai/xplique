# Prototypes
Prototype-based explanation is a family of natural example-based XAI methods. Prototypes consist of a set of samples that are representative of either the dataset or a class. Three classes of prototype-based methods are found in the literature ([Poché et al., 2023](https://hal.science/hal-04117520/document)): 

- [Prototypes for Data-Centric Interpretability](#prototypes-for-data-centric-interpretability)
- [Prototypes for Post-hoc Interpretability](#prototypes-for-post-hoc-interpretability)
- Prototype-Based Models Interpretable by Design

For now, the library focuses on the first two classes.

## Common API ##

```python

explainer = Method(cases_dataset, labels_dataset, targets_dataset, k, 
                   projection, case_returns, batch_size, distance, 
                   nb_prototypes, kernel_type, 
                   kernel_fn, gamma)
# compute global explanation
global_prototypes = explainer.get_global_prototypes()
# compute local explanation
local_prototypes = explainer(inputs)

```

??? abstract "Table of methods available"

    The following Data-Centric prototypes methods are implemented:

    | Method Name and Documentation link     | **Tutorial**             | Available with TF | Available with PyTorch* |
    |:-------------------------------------- | :----------------------: | :---------------: | :---------------------: |
    | [ProtoGreedy](../proto_greedy/)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
    | [ProtoDash](../proto_dash/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
    | [MMDCritic](../mmd_critic/)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |

    *: Before using a PyTorch model it is highly recommended to read the [dedicated documentation](../pytorch/)

!!!info
    Using the identity projection, one is looking for the **dataset prototypes**. In contrast, using the latent space of a model as a projection, one is looking for **prototypes relevant for the model**.

!!!info
    Prototypes, share a common API with other example-based methods. Thus, to understand some parameters, we recommend reading the [dedicated documentation](../../api_example_based/).

## Prototypes for Data-Centric Interpretability

In this class, prototypes are selected without relying on the model and provide an overview of
the dataset. As mentioned in ([Poché et al., 2023](https://hal.science/hal-04117520/document)), we found in this class: **clustering methods** and **data summarization methods**, also known as **set cover methods**. This library focuses on **data summarization methods** which can be treated in two ways [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf): 

- **Data summarization with knapsack constraint**: 
consists in finding a subset of prototypes $\mathcal{P}$ that maximizes the coverage set function $F(\mathcal{P})$ under the constraint that its selection cost $C(\mathcal{P})$ (e.g., the number of selected prototypes $|\mathcal{P}|$) should be less than a given budget. 

- **Data summarization with covering constraint**:
consists in finding a low-cost subset  of prototypes $\mathcal{P}$ under the constraint it should cover all the data. 

For both cases, submodularity and monotonicity of $F(\mathcal{P})$ are necessary to guarantee that a greedy algorithm has a constant factor guarantee of optimality [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf). In addition, $F(\mathcal{P})$ should encourage coverage and penalize redundancy in order to have a good summary [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf).

The library implements three methods from **Data summarization with knapsack constraint**: `MMDCritic`, `ProtoGreedy` and `ProtoDash`.

[Kim et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf) proposed `MMDCritic` method that used a set function based on the Maximum Mean Discrepancy [(MMD)](#what-is-mmd). They solved **data summarization with knapsack constraint** problem to find both prototypes and criticisms. First, the number of prototypes and criticisms to be found, respectively as $m_p$ and $m_c$, are selected. Second, to find prototypes, a greedy algorithm is used to maximize $F(\mathcal{P})$ s.t. $|\mathcal{P}| \le m_p$ where $F(\mathcal{P})$ is defined as:

\begin{equation}
    F(\mathcal{P})=\frac{2}{|\mathcal{P}|\cdot n}\sum_{i,j=1}^{|\mathcal{P}|,n}\kappa(p_i,x_j)-\frac{1}{|\mathcal{P}|^2}\sum_{i,j=1}^{|\mathcal{P}|}\kappa(p_i,p_j)
\end{equation}

They used diagonal dominance conditions on the kernel to ensure monotonocity and submodularity of $F(\mathcal{P})$. To find criticisms $\mathcal{C}$, the same greedy algorithm is used to select points that maximize another objective function $J(\mathcal{C})$. 

[Gurumoorthy et al., 2019](https://arxiv.org/pdf/1707.01212) associated non-negative weights to prototypes which are indicative of their importance. This approach allows for identifying both prototypes and criticisms (the least weighted examples among prototypes) by maximizing the same weighted objective $F(\mathcal{P},w)$ defined as:

\begin{equation}   
    F(\mathcal{P},w)=\frac{2}{n}\sum_{i,j=1}^{|\mathcal{P}|,n}w_i\kappa(p_i,x_j)-\sum_{i,j=1}^{|\mathcal{P}|}w_iw_j\kappa(p_i,p_j),
\end{equation}

where $w$ are non-negative weights for each prototype. The problem then consist on finding $\mathcal{P}$ with a corresponding $w$ that maximizes $J(\mathcal{P}) \equiv \max_{w:supp(w)\in \mathcal{P},w\ge 0} J(\mathcal{P},w)$ s.t. $|\mathcal{P}| \leq m=m_p+m_c$. They established the weak submodular property of $J(\mathcal{P})$ and present tractable algorithms (`ProtoGreedy` and `ProtoDash`) to optimize it. 

### Method comparison

- Compared to `MMDCritic`, both `ProtoGreedy` and `Protodash` additionally determine the weights for each of the selected prototypes. 
- `ProtoGreedy` and `Protodash` works for any symmetric positive definite kernel which is not the case for `MMDCritic`. 
- `MMDCritic` and `ProtoGreedy` select the next element that maximizes the increment of the scoring function while `Protodash` maximizes a tight lower bound on the increment of the scoring function (it maximizes the gradient of $F(\mathcal{P},w)$).
- `ProtoDash` is much faster than `ProtoGreedy` without compromising on the quality of the solution (the complexity of `ProtoGreedy` is $O(n(n+m^4))$ comparing to $O(n(n+m^2)+m^4)$ for `ProtoDash`). 
- The approximation guarantee for `ProtoGreedy` is $(1-e^{-\gamma})$, where $\gamma$ is submodularity ratio of $F(\mathcal{P})$, comparing to $(1-e^{-1})$ for `MMDCritic`.

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
    For `MMDCritic`, the kernel must satisfy a condition ensuring the submodularity of the set function (the Gaussian kernel respects this constraint). In contrast, for `ProtoDash` and `ProtoGreedy`, any kernel can be used, as these methods rely on weak submodularity instead of full submodularity.

### Default kernel
The default kernel used is Gaussian kernel. This kernel distance assigns higher similarity to points that are close in feature space and gradually decreases similarity as points move further apart. It is a good choice when your data has complexity. However, it can be sensitive to the choice of hyperparameters, such as the width $\sigma$ of the Gaussian kernel, which may need to be carefully fine-tuned.

### Implementation details

The search method for `ProtoGreedy` inherits from the `BaseSearchMethod` class. It finds prototypes and assigns a non-negative weight to each one.

Both the search methods for `MMDCritic` and `ProtoDash` classes inherit from the one defined for `ProtoGreedy`. The search method for `MMDCritic` differs from `ProtoGreedy` by assigning equal weights to the selection of prototypes. The two classes use the same greedy algorithm. In the `compute_objective` method of the search method of `ProtoGreedy`, for each new candidate, we calculate the best weights for the selection of prototypes. However, in `MMDCritic`, the `compute_objective` method assigns the same weight to all elements in the selection.

`ProtoDash`, like `ProtoGreedy`, assigns a non-negative weight to each prototype. However, the algorithm used by `ProtoDash` is [different](#method-comparison) from the one used by `ProtoGreedy`. Therefore, search method of `ProtoDash` overrides both the `compute_objective` method and the `update_selection` method.

## Prototypes for Post-hoc Interpretability

Data-Centric methods such as `ProtoGreedy`, `ProtoDash` and `MMDCritic` can be used in either the output or the latent space of the model. In these cases, [projections methods](../../projections/) are used to transfer the data from the input space to the latent/output spaces.
