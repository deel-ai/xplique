# Prototypes
A prototype in AI explainability is a representative example from the data that shows how the model makes decisions ([Poché et al., 2023](https://hal.science/hal-04117520/document)). It helps explain a prediction by pointing to a similar example the model learned from, making the decision more understandable. Imagine you're training a model to recognize dogs. After the model learns, you can ask it to show a "prototype" for the dog category, which would be an actual image from the training set that best represents what a typical dog looks like.

!!!info
    Using the identity projection, one is looking for the **dataset prototypes**. In contrast, using the latent space of a model as a projection, one is looking for **prototypes relevant for the model**.

## Common API ##

```python

explainer = Method(cases_dataset, labels_dataset, nb_local_prototypes, projection, 
                   case_returns, batch_size, distance, nb_global_prototypes)
# compute global explanation
global_prototypes = explainer.get_global_prototypes()
# compute local explanation
local_prototypes = explainer(inputs)

```

??? abstract "Table of methods available"

    The following prototypes methods are implemented:

    | Method Name and Documentation link     | **Tutorial**             | Available with TF | Available with PyTorch* |
    |:-------------------------------------- | :----------------------: | :---------------: | :---------------------: |
    | [ProtoGreedy](../proto_greedy/)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
    | [ProtoDash](../proto_dash/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |
    | [MMDCritic](../mmd_critic/)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) | ✔ | ✔ |

!!!info
    Prototypes, share a common API with other example-based methods. Thus, to understand some parameters, we recommend reading the [dedicated documentation](../../api_example_based/).

## Specificity of prototypes
 
The search method class related to a `Prototypes` class includes the following additional parameters:  

- `nb_global_prototypes` which represents the total number of prototypes desired to represent the entire dataset. 
- `nb_local_prototypes` which represents the number of prototypes closest to the input and allows for a local explanation. This attribute is equivalent to $k$ in the other exemple based methods.

- `kernel_type`, `kernel_fn`, and `gamma` which are related to the [kernel](#how-to-choose-the-kernel) used to compute the [MMD distance](#what-is-mmd).

The prototype class has a `get_global_prototypes()` method, which calculates all the prototypes in the base dataset; these are called the global prototypes. The `explain` method then provides a local explanation, i.e., finds the prototypes closest to the input given as a parameter.

## Implemented methods

The library implements three methods, `MMDCritic`, `ProtoGreedy` and `ProtoDash` from **Data summarization with knapsack constraint** [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf). This class of prototype methods involves finding a subset of prototypes $\mathcal{P}$ that maximizes the coverage set function $F(\mathcal{P})$ under the constraint that its selection cost $C(\mathcal{P})$ (e.g., the number of selected prototypes $|\mathcal{P}|= nb\_global\_prototypes$) should be less than a given budget. 
Submodularity and monotonicity of $F(\mathcal{P})$ are necessary to guarantee that a greedy algorithm has a constant factor guarantee of optimality [(Lin et al., 2011)](https://aclanthology.org/P11-1052.pdf).

### Method comparison

- Compared to `MMDCritic`, both `ProtoGreedy` and `Protodash` additionally determine the weights for each of the selected prototypes. 
- `ProtoGreedy` and `Protodash` works for any symmetric positive definite kernel which is not the case for `MMDCritic`. 
- `MMDCritic` and `ProtoGreedy` select the next element that maximizes the increment of the scoring function while `Protodash` maximizes a tight lower bound on the increment of the scoring function (it maximizes the gradient of $F(\mathcal{P},w)$).
- `ProtoDash` is much faster than `ProtoGreedy` without compromising on the quality of the solution. The complexity of `ProtoGreedy` is $O(n(n+m^4))$ comparing to $O(n(n+m^2)+m^4)$ for `ProtoDash`. 
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

!!!info
    The default kernel used is Gaussian kernel. This kernel distance assigns higher similarity to points that are close in feature space and gradually decreases similarity as points move further apart. It is a good choice when your data has complexity. However, it can be sensitive to the choice of hyperparameters, such as the width $\sigma$ of the Gaussian kernel, which may need to be carefully fine-tuned.


