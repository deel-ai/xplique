# MMDCritic

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/antonin/example-based-merge/xplique/example_based/search_methods/proto_greedy_search.py) |
📰 [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf)

`MMDCritic` finds prototypes and criticisms by maximizing two separate objectives based on the Maximum Mean Discrepancy (MMD).

!!! quote
    MMD-critic uses the MMD statistic as a measure of similarity between points and potential prototypes, and
    efficiently selects prototypes that maximize the statistic. In addition to prototypes, MMD-critic selects criticism samples i.e. samples that are not well-explained by the prototypes using a regularized witness function score.

    -- <cite>[Efficient Data Representation by Selecting Prototypes with Importance Weights (2019).](https://arxiv.org/abs/1707.01212)</cite>

First, to find prototypes $\mathcal{P}$, a greedy algorithm is used to maximize $F(\mathcal{P})$ s.t. $|\mathcal{P}| \le m_p$ where $F(\mathcal{P})$ is defined as:
\begin{equation}
    F(\mathcal{P})=\frac{2}{|\mathcal{P}|\cdot n}\sum_{i,j=1}^{|\mathcal{P}|,n}\kappa(p_i,x_j)-\frac{1}{|\mathcal{P}|^2}\sum_{i,j=1}^{|\mathcal{P}|}\kappa(p_i,p_j),
\end{equation}
where $m_p$ the number of prototypes to be found. They used diagonal dominance conditions on the kernel to ensure monotonocity and submodularity of $F(\mathcal{P})$. 

Second, to find criticisms $\mathcal{C}$, the same greedy algorithm is used to select points that maximize another objective function $J(\mathcal{C})$. 

!!!warning
    For `MMDCritic`, the kernel must satisfy a condition that ensures the submodularity of the set function. The Gaussian kernel meets this requirement and it is recommended. If you wish to choose a different kernel, it must satisfy the condition described by [Kim et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf).


## Example

```python
from xplique.example_based import MMDCritic

# load data and labels
# ...

explainer = MMDCritic(cases_dataset, labels_dataset, targets_dataset, k, 
                   projection, case_returns, batch_size, distance, 
                   nb_prototypes, kernel_type, 
                   kernel_fn, gamma)
# compute global explanation
global_prototypes = explainer.get_global_prototypes()
# compute local explanation
local_prototypes = explainer(inputs)
```

## Notebooks

- [**Prototypes**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**MMDCritic**: Going Further](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X)


{{xplique.example_based.prototypes.MMDCritic}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)

