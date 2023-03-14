# API: Metrics

## Context

As the XAI field continues on being trendy, the quantity of materials at disposal to explain DL models keeps on growing. Especially, there is an increasing need to benchmark and evaluate those different approaches. Mainly, there is an urge to evaluate the quality of explanations provided by attribution methods.

!!!info
    Note that, even though some work exists for other tasks, this challenge has been mainly tackled in the context of Computer Vision tasks.

As pointed out by [Petsiuk et al.](http://arxiv.org/abs/1806.07421) most explanations approaches are used to be evaluated in a human-centred way.  For instance, an attribution method was considered as good if it pointed out the same relevant pixels as the ones highlighted by human users. While this kind of evaluation allows giving some user trust it can easily be biased. Therefore, the authors introduced two automatic evaluation metrics that rely solely on the drop or rise in the probability of a class as important pixels (defined by the saliency map) are removed or added. Those are not the only available metrics and we propose here to present the API we used as common ground and then to dive into more specifity.

## Common API

All metrics inherits from the base class `BaseAttributionMetric` which has the following `__init__` arguments:

-   \- `model`: The model from which we want to obtain explanations
-   \- `inputs`: Input samples to be explained

    !!!warning
        Inputs should be the same as defined in the attribution's [API Description](https://deel-ai.github.io/xplique/api/attributions/api_attributions/)

-   \- `targets`: One-hot encoding of the model's output from which an explanation is desired

    !!!warning
        Idem

-   \- `batch_size`

Then we can distinguish two category of metrics:

-   \- Those which only need the attribution ouputs of an explainer: `ExplanationMetric`
-   \- Those which need the explainer: `ExplainerMetric`

### `ExplanationMetric`

Those metrics are agnostic of the explainer used and rely only on the attributions mappings it gives.

!!!tip
    Therefore, you can use them with other explainer than those provided in Xplique!

All metrics inheriting from this class have another argument in their `__init__` method:

-   \- `operator`: Optionnal function wrapping the model. It can be seen as a metric which allow to evaluate model evolution. For more details, see the attribution's [API Description](https://deel-ai.github.io/xplique/api/attributions/api_attributions/)

All metrics inheriting from this class have to define a method `evaluate` which will take as input the `attributions` given by an explainer. Those attributions should correspond to the `model`, `inputs` and `targets` used to build the metric object.

Especially, all Fidelity metrics inherit from this class:

| Metric Name (Fidelity) |Notebook                                                                                                                                                           |
|:---------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MuFidelity             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nuqLezSHavXGMsGtHrdSajEcR1SCzqTA) |
| Insertion              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QtwbegOpTSj7g6DxBprMt0aTtaV5surF) |
| Deletion               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W7tfXOoPnbu4HGGIkbhkoKdk9xRdStgs) |


### `ExplainerMetric`

Those metrics will not assess the quality of the explanations provided but (also) the explainer itself.

All metrics inheriting from this class have to define a method `evaluate` which will take as input the `explainer` evaluated.

!!!info
    It is even more important that `inputs` and `targets` are the same as defined in the attribution's [API Description](https://deel-ai.github.io/xplique/api/attributions/api_attributions/)

Currently, there is only one Stability metric inheriting from this class:

| Metric Name (Stability) |Notebook                                                                                                                                                           |
|:----------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AverageStability        | **(WIP)**                                                                                                                                                                 |

## Other Metrics

A Representatibity metric: [MeGe](https://arxiv.org/abs/2009.04521) is also available. Documentation about it should be added soon.

## Notebooks

- [**Metrics**: Getting started](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) </sub>
