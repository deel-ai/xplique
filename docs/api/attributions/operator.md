# Operator

`operator` is one of the main parameters for both attribution methods and metrics. It defines the function $g$ that we want to explain. *E.g.*: In the case we have a classifier model the function that we might want to explain is the one that given a target gives us the score of the model for that specific target -- *i.e* $model(input)[target]$.

!!!note
    The `operator` parameter is a feature avaible for version > $1.$. The `operator` default values are the ones used before the introduction of this new feature!

## Leitmotiv

The `operator` parameter was introduced to offer users a flexible way to adapt current attribution methods or metrics. It should help them to empirically tackle new use-cases/new tasks. Broadly speaking, it should amplify the user's ability to experiment. However, this also imply that it is the user responsability to make sure that its derivations are in-scope of the original method and make sense.

## Operators' Signature

An `operator` is a function $g$ that we want to explain. This function take as input $3$ parameters:

- `model`, the model under investigation
- `inputs`: One of the following: a `tf.data.Dataset` (in which case you should not provide `targets`), a `tf.Tensor` or a `np.ndarray`
- `targets`: One of the following: a `tf.Tensor` or a `np.ndarray`

!!!info
    More specification concerning `model` or `inputs` can be found in the [model's documentation](../model/). More information on `targets` can be found [here](#tasks) or also in the [model's documentation](../model/#tasks)

This function $g$ should return a **vector of scalar value** of size $(N,)$ where $N$ is the number of input in `inputs` -- *i.e* a scalar score per input.

## How is the operator used in Xplique ?

### Black-box attribution methods

For attribution approaches that do not require gradient computation we mostly need to query the model. Thus, those methods need an inference function. If you provide an `operator`, it will be the inference function.

More concretely, for this kind of approach you want to compare some valued function for an original input and perturbed version of it:

```python
original_scores = operator(model, original_inputs, original_targets)

# depending on the attribution method this `perturbation_function` is different
perturbed_inputs, perturbed_targets = perturbation_function(original_inputs, original_targets)
perturbed_scores = operator(model, perturbed_inputs, perturbed_targets)

# exemple of comparison of interest
diff_scores = math.sqrt((original_scores - perturbed_scores)**2)
```

### White-box attribution methods

Those methods usually require some gradients computation. The gradients that will be used are the one of the operator function (see the `get_gradient_of_operator` method in the [Providing custom operator](#providing-custom-operator) section). 

## Default Behavior

A lot of attribution methods are initially intended for classification tasks. Thus, the default operator `predictions_operator` assume such a setting

```python
@tf.function
def predictions_operator(model: Callable,
                         inputs: tf.Tensor,
                         targets: tf.Tensor) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels, one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
    return scores
```

That is a setting where the variable `model(inputs)` is a vector of size $(N, C)$ where: $N$ is the number of input and $C$ is the number of class.

!!!info
    Explaining the logits is to explain the class, while explaining the softmax is to explain why this class is more likely. Thus, it is recommended to explain the logit and exclude the softmax layer if any.

## Existing operators and how to use them

At present, there are at present 2 operators available (and 2 others should be released soon) in the library that tackle different tasks.

### Tasks

#### Classification Tasks

!!!tip
    In general, if you are doing classification tasks it is better to not include the final softmax layer in your model but to work with logits instead!

For classification tasks, it is expected for the user to use the `predictions_operator` when initializing an explainer:

```python
@tf.function
def predictions_operator(model: Callable,
                         inputs: tf.Tensor,
                         targets: tf.Tensor) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
    return scores
```

- `model(inputs)`
Consequently, we expect `model(inputs)` to yield a $(N, C)$ tensor or array where $N$ is the number of input samples and $C$ is the number of classes. 

- `targets`
If you use the default operator for classification task we expect `targets` to be a $(N, C)$ tensor or array which is a one-hot encoding of **the class you want to explain** where $N$ is the number of input samples and $C$ is the number of classes.

#### Regression Tasks

If the task at end is regression, then the user should instantiate the explainer with the `regression_operator`:

```python
@tf.function
def regression_operator(model: Callable,
                        inputs: tf.Tensor,
                        targets: tf.Tensor) -> tf.Tensor:
    """
    Compute the the mean absolute error between model prediction and the target.
    Target should the model prediction on non-perturbed input.
    This operator can be used to compute attributions for all outputs of a regression model.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        Model prediction on non-perturbed inputs.

    Returns
    -------
    scores
        MAE between model prediction and targets.
    """
    scores = tf.reduce_mean(tf.abs(model(inputs) - targets), axis=-1)
    return scores
```

- `model(inputs)`:
Consequently, we expect `model(inputs)` to yield a $(N, D)$ tensor or array where $N$ is the number of input samples and $D$ is the number of variables the model should predict (possibly one). 

- `targets`:
If you are using the `regression_operator`, it is expected that `targets` to be a $(N, D)$ tensor or array of the expected multi-variate output where $N$ is the number of input samples and $D$ is the number of variables (possibly one).

#### Object-Detection Tasks

**Work In Progress**

#### Segmentation Tasks

**Work In Progress**

### How to use them with an explainer ?

You can build attribution methods with those operator in three ways:

- Explicitly importing them

```python
from xplique.attributions import Saliency
from xplique.commons.operators import regression_operator

explainer = Saliency(model, operator=regression_operator)
explanations = explainer(inputs, targets)
```

At present, the available operators are: `predictions_operator` and `regression_operator`

- Use their name

```python
from xplique.attributions import Saliency

explainer = Saliency(model, operator='regression')
explanations = explainer(inputs, targets)
```

At present you can select a name in ["classification", "regression"]

- Use the `Tasks` enumeration

```python
from xplique.commons import Tasks
from xplique.attributions import Saliency

explainer = Saliency(model, operator=Tasks.REGRESSION)
explanations = explainer(inputs, targets)
```

At present the `Tasks` enum has two members: `CLASSIFICATION` and `REGRESSION`

## Providing custom operator

If you provide a custom operator you should be aware that:

- An assertion will be made to ensure it respects the signature describe in the previous section
- Your operator will go through the `get_gradient_of_operator` method if you use any white-box explainer

```python
def get_gradient_of_operator(operator):
    """
    Get the gradient of an operator.

    Parameters
    ----------
    operator
        Operator to compute the gradient of.

    Returns
    -------
    gradient
        Gradient of the operator.
    """
    @tf.function
    def gradient(model, inputs, targets):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            scores = operator(model, inputs, targets)

        return tape.gradient(scores, inputs)

    return gradient
```

!!!tip
    Writing your operator with only tensorflow functions should increase your chance that this method does not yield any errors.

!!!warning
    Note that depending on your operator, the targets you provide should make sense

## Examples of applications

**WIP**

