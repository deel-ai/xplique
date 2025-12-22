# API: Metrics

- [**Attribution Methods**: Metrics](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) </sub>





## Context

As the XAI field continues on being trendy, the quantity of materials at disposal to explain DL models keeps on growing. Especially, there is an increasing need to benchmark and evaluate those different approaches. Mainly, there is an urge to evaluate the quality of explanations provided by attribution methods.

!!!info
    Note that, even though some work exists for other tasks, this challenge has been mainly tackled in the context of Computer Vision tasks.

As pointed out by [Petsiuk et al.](http://arxiv.org/abs/1806.07421), most explainability approaches used to be evaluated in a human-centered way. For instance, an attribution method was considered good if it pointed at the same relevant pixels as the ones highlighted by human users. While this kind of evaluation allows giving some users trust, it can easily be biased. Therefore, the authors introduced two automatic evaluation metrics that rely solely on the drop or rise in the probability of a class as important pixels (defined by the saliency map) are removed or added. Those are not the only available metrics and we propose here to present the API we used as common ground.





## Common API

!!!info
    Metrics described on this page are metrics for attribution methods and explanations. Therefore, the user should first get familiar with the [attributions methods API](../../api_attributions/) as many parameters are common between both API. For instance, `model`, `inputs`, `targets`, and `operator` should match for methods and their metrics.

All metrics inherit from the base class `BaseAttributionMetric` which has the following `__init__` arguments:

- `model`: The model from which we want to obtain explanations
- `inputs`: Input samples to be explained
- `targets`: Specify the kind of explanations we want depending on the task at end (e.g. a one-hot encoding of a class of interest, a difference to a ground-truth value..)
- `batch_size`: an integer which allows to either process inputs per batch or process perturbed samples of an input per batch (inputs are therefore processed one by one). It is most of the time overwritten by the explanation method `batch_size`.
- `activation`: A string that belongs to [None, 'sigmoid', 'softmax']. See the [dedicated section](#activation) for details

Then we can distinguish two category of metrics:

- Those which only need the attribution outputs of an explainer: `ExplanationMetric`, namely:
    - **Fidelity metrics**: [MuFidelity](../mu_fidelity), [Deletion](../deletion), [Insertion](../insertion), [AverageDropMetric](../average_drop), [AverageGainMetric](../average_gain), [AverageIncreaseMetric](../average_increase)
    - **Complexity metrics**: [Complexity](../complexity), [Sparseness](../sparseness)
- Those which need the explainer: `ExplainerMetric`:
    - **Stability metrics**: [AverageStability](../avg_stability)
    - **Randomization metrics**: [RandomLogitMetric](../random_logit), [ModelRandomizationMetric](../model_randomization)




### `ExplanationMetric`

Those metrics are agnostic of the explainer used and rely only on the attributions mappings it gives.

!!!tip
    Therefore, you can use them with other explainer than those provided in Xplique!

All metrics inheriting from this class have another argument in their `__init__` method:

- `operator`: Optional function wrapping the model. It can be seen as a metric which allows to evaluate model evolution. For more details, see the attribution's [API Description section on `operator`](../../api_attributions/#tasks-and-operator).

!!!info
    The `operator` used here should match the one used to compute the explanations!

All metrics inheriting from this class have to define a method `evaluate` which will take as input the `attributions` given by an explainer. Those attributions should correspond to the `model`, `inputs` and `targets` used to build the metric object.



### `ExplainerMetric`

These metrics will not assess the quality of the explanations provided but (also) the explainer itself.

All metrics inheriting from this class have to define a method `evaluate` which will take as input the `explainer` evaluated.

!!!info
    It is even more important that `inputs` and `targets` be the same as defined in the attribution's [API Description](../../api_attributions/#inputs).

Currently, there is only one Stability metric inheriting from this class:






## Activation

This parameter specifies if an additional activation layer should be added once a model has been called on the inputs when you have to compute the metric. 

Indeed, most of the times it is recommended when you instantiate an **explainer** (*i.e.* an attribution methods) to provide a model which gives logits as explaining the logits is to explain the class, while explaining the softmax is to explain why this class rather than another.

However, when you compute metrics some were thought to measure a "drop of probability" when you occlude the "most relevant" part of an input. Thus, once you get your explanations (computed from the logits), you might need to have access to a probability score of occluded inputs of a specific class, thus to have access to the logits after a `softmax` or `sigmoid` layer.

Consequently, we add this `activation` parameter so one can provide a model that predicts logits but add an activation layer for the purpose of having probability when using a metric method.

The default behavior is to compute the metric without adding any activation layer to the model.

!!!note
    There does not appear to be a consensus on the activation function to be used for metrics. Some papers use logits values (e.g., with mu-fidelity), while others use sigmoid or softmax (with deletion and insertion). We can only observe that changing the activation function has an effect on the ranking of the best methods.
 




## Other Fidelity Metrics

Other fidelity metrics that are much less computationally expensive:

- [**AverageDropMetric**](../average_drop): Measures the relative drop in confidence when masking inputs with the explanation.
- [**AverageGainMetric**](../average_gain): Measures the relative increase in confidence (complementary to Average Drop).
- [**AverageIncreaseMetric**](../average_increase): Binary indicator for whether masking increases confidence.




## Complexity Metrics

These metrics evaluate the interpretability of explanations based on their structure and complexity:

- [**Complexity**](../complexity): Entropy-based measure of how diffuse/concentrated explanations are.
- [**Sparseness**](../sparseness): Gini-index-based measure of attribution concentration.




## Randomization Metrics (Sanity Checks)

These metrics implement sanity checks to verify that explainers are sensitive to the model and target labels:

- [**RandomLogitMetric**](../random_logit): Tests whether explanations change when the target class is randomized.
- [**ModelRandomizationMetric**](../model_randomization): Tests whether explanations degrade when model parameters are randomized.

!!!tip
    These metrics are based on the sanity checks proposed by [Adebayo et al. (2018)](https://arxiv.org/abs/1810.03292). Low similarity scores indicate faithful explainers.




## Other Metrics

A Representativity metric: [MeGe](https://arxiv.org/abs/2009.04521) is also available. Documentation about it should be added soon.





## Notebooks

- [**Metrics**: Getting started](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) </sub>

- [**Metrics**: With PyTorch models](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) </sub>
