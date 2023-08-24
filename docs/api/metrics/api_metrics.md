# API: Metrics

## Context

As the XAI field continues on being trendy, the quantity of materials at disposal to explain DL models keeps on growing. Especially, there is an increasing need to benchmark and evaluate those different approaches. Mainly, there is an urge to evaluate the quality of explanations provided by attribution methods.

!!!info
    Note that, even though some work exists for other tasks, this challenge has been mainly tackled in the context of Computer Vision tasks.

As pointed out by [Petsiuk et al.](http://arxiv.org/abs/1806.07421) most explanations approaches are used to be evaluated in a human-centred way.  For instance, an attribution method was considered as good if it pointed out the same relevant pixels as the ones highlighted by human users. While this kind of evaluation allows giving some user trust it can easily be biased. Therefore, the authors introduced two automatic evaluation metrics that rely solely on the drop or rise in the probability of a class as important pixels (defined by the saliency map) are removed or added. Those are not the only available metrics and we propose here to present the API we used as common ground and then to dive into more specifity.

## Common API

All metrics inherits from the base class `BaseAttributionMetric` which has the following `__init__` arguments:

- `model`: The model from which we want to obtain explanations
- `inputs`: Input samples to be explained

    !!!info
        Inputs should be the same as defined in the [model's documentation](../../attributions/model)

- `targets`: Specify the kind of explanations we want depending on the task at end (e.g. a one-hot encoding of a class of interest, a difference to a ground-truth value..)

    !!!info
        Targets should be the same as defined in the [model's documentation](../../attributions/model)

-  `batch_size`

-  `activation`: A string that belongs to [None, 'sigmoid', 'softmax']. See the [dedicated section](#activation) for details

Then we can distinguish two category of metrics:

- Those which only need the attribution ouputs of an explainer: `ExplanationMetric`
- Those which need the explainer: `ExplainerMetric`

### `ExplanationMetric`

Those metrics are agnostic of the explainer used and rely only on the attributions mappings it gives.

!!!tip
    Therefore, you can use them with other explainer than those provided in Xplique!

All metrics inheriting from this class have another argument in their `__init__` method:

- `operator`: Optionnal function wrapping the model. It can be seen as a metric which allow to evaluate model evolution. For more details, see the attribution's [API Description](../../attributions/api_attributions/) and the [operator documentation](../../attributions/operator/)

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
    It is even more important that `inputs` and `targets` are the same as defined in the attribution's [API Description](../../attributions/api_attributions/)

Currently, there is only one Stability metric inheriting from this class:

| Metric Name (Stability) |Notebook                                                                                                                                                           |
|:----------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AverageStability        | **(WIP)**                                                                                                                                                                 |

## Activation

This parameter specify if an additional activation layer should be added once a model has been called on the inputs when you have to compute the metric. 

Indeed, most of the times it is recommended when you instantiate an **explainer** (*i.e.* an attribution methods) to provide a model which gives logits as explaining the logits is to explain the class, while explaining the softmax is to explain why this class rather than another.

However, when you compute metrics some were thought to measure a "drop of probability" when you occlude the "most relevant" part of an input. Thus, once you get your explanations (computed from the logits), you might need to have access to a probability score of occluded inputs of a specific class, thus to have access to the logits after a `softmax` or `sigmoid` layer.

Consequently, we add this `activation` parameter so one can provide a model that predicts logits but add an activation layer for the purpose of having probability when using a metric method.

The default behavior is to compute the metric without adding any activation layer to the model.

!!!note
    In our opinion, there is no consensus at present concerning the "best practices". Should the model used to generate the explanations be exactly the same for generating the metrics or it should depend on the metric ? As we do not claim to have an answer (yet!), we choose to let the user as much flexibility as possible!
 
## Other Metrics

A Representatibity metric: [MeGe](https://arxiv.org/abs/2009.04521) is also available. Documentation about it should be added soon.

## Notebooks

- [**Metrics**: Getting started](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) </sub>

- [**Metrics**: With Pytorch's model](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) </sub>
