# Sobol Attribution Method

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) | 
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/global_sensitivity_analysis/sobol_attribution_method.py) |
📰 [Paper](https://arxiv.org/pdf/2111.04138.pdf)

The Sobol attribution method from Fel, Cadène & al.[^1] is an attribution method grounded in Sensitivity Analysis.
Beyond modeling the individual contributions of image regions, Sobol indices provide
an efficient way to capture higher-order interactions between image regions and their
contributions to a neural network’s prediction through the lens of variance.

!!! quote
    The total Sobol index $ST_i$ which measures the contribution
    of the variable $X_i$ as well as its interactions of any order with any other input variables to the model
    output variance.

    -- <cite>[Look at the Variance! Efficient Black-box Explanations with Sobol-based Sensitivity Analysis (2021)](https://arxiv.org/pdf/2111.04138.pdf)</cite>[^1]

More precisely, the attribution score $\phi_i$ for an input variable $x_i$, is defined as

$$ \phi_i = \frac{\mathbb{E}_{X \sim i}(Var_{X_i}(f(x) | X_{\sim i}))} {Var
(f(X
))} $$

Where $\mathbb{E}_{X \sim i}(Var_{X_i}(f(x) | X_{\sim i}))$ is the expected variance
that would be left if all variables but $X_{\sim i}$ were to be fixed.


In order to generate stochasticity($X_i$), a perturbation function is used and uses perturbation masks
to modulate the generated perturbation. The perturbation functions available are inpainting
that modulates pixel regions to a baseline state, amplitude and blurring.

The calculation of the indices also requires an estimator -- in practice this parameter does not
change the results much -- `JansenEstimator` being recommended. 

Finally the exploration of the manifold exploration is made using a sampling method, several samplers are proposed: Quasi-Monte
Carlo (`ScipySobolSequence`, recommended) using Scipy's sobol sequence, Latin hypercubes
 -- `LHSAmpler` -- or Halton's sequences `HaltonSequence`.


!!!tip
    For quick a faithful explanations, we recommend to use `grid_size` in $[7, 12)$,
    `nb_design` in $\{16, 32, 64\}$ (more is useless), and a QMC sampler.
    (see `SobolAttributionMethod` documentation below for detail on those parameters).

## Example

```python
from xplique.attributions import SobolAttributionMethod
from xplique.attributions.global_sensitivity_analysis import (
    JansenEstimator, GlenEstimator,
    LHSampler, ScipySobolSequence,
    HaltonSequence)

# load images, labels and model
# ...

# default explainer (recommended)
explainer = SobolAttributionMethod(model, grid_size=8, nb_design=32)
explanations = method(images, labels) # one-hot encoded labels
```

If you want to change the estimator or the sampling:

```python
from xplique.attributions import SobolAttributionMethod
from xplique.attributions.global_sensitivity_analysis import (
    JansenEstimator, GlenEstimator,
    LHSampler, ScipySobolSequence,
    HaltonSequence)

# load images, labels and model
# ...

explainer_lhs = SobolAttributionMethod(model, grid_size=8, nb_design=32, 
                                       sampler=LHSampler(), 
                                       estimator=GlenEstimator())
explanations_lhs = explainer_lhs(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>


{{xplique.attributions.global_sensitivity_analysis.sobol_attribution_method.SobolAttributionMethod}}

[^1]:[Look at the Variance! Efficient Black-box Explanations with Sobol-based Sensitivity Analysis (2021)](https://arxiv.org/pdf/2111.04138.pdf)
