# Operator

`operator` is one of the main parameters for both attribution methods and metrics. It defines the function $g$ that we want to explain. *E.g.*: In the case we have a classifier model the function that we might want to explain is the one that given a target gives us the score of the model for that specific target -- *i.e* $model(input)[target]$.

!!!note
    The `operator` parameter is a feature avaible for version > $1.$. The `operator` default values are the ones used before the introduction of this new feature!

## Leitmotiv

The `operator` parameter was introduced to offer users a flexible way to adapt current attribution methods or metrics. It should help them to empirically tackle new use-cases/new tasks. Broadly speaking, it should amplify the user's ability to experiment. However, this also imply that it is the user responsability to make sure that its derivationns are in-scope of the original method and make sense.  

## Operators' Signature

An `operator` is a function $g$ that we want to explain. This function take as input $3$ parameters:

- `model`, the model under investigation
- `inputs`: One of the following: a `tf.data.Dataset` (in which case you should not provide `targets`), a `tf.Tensor` or a `np.ndarray`
- `targets`: One of the following: a `tf.Tensor` or a `np.ndarray`

!!!info
    More specification concerning `model` or `inputs` can be found in the [model's documentation](../model/)

This function $g$ should return a **vector of scalar value** of size $(N,)$ where $N$ is the number of input in `inputs` -- *i.e* a scalar score per input.

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

## How is the operator used in Xplique ?

### Black-box attribution methods

For attribution approaches that do not require gradient computation we mostly need to query the model. Thus, those methods need an inference function. If you provide an `operator`, it will be the inference function.

More concretely, for this kind of approach you want to compare some valued function for an original input and perturbed version of it:

```python
original_scores = operator(model, original_inputs, original_targets)

# depending on the attribution method this `perturbation_function is different`
perturbed_inputs, perturbed_targets = perturbation_function(original_inputs, original_targets)
perturbed_scores = operator(model, perturbed_inputs, perturbed_targets)

# exemple of comparison of interest
diff_scores = math.sqrt((original_scores - perturbed_scores)**2)
```

### White-box attribution methods

Those methods usually require some gradients computation. The gradients that will be used are the one of the operator function (see the `get_gradient_of_operator` method in the previous section). 

## Default Behavior

### Attribution methods

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

### Metrics

It is recommended when one initialize a metric to use the same `operator` than the one used for the attribution methods. **HOWEVER** it should be pointed out that the default behavior **add a softmax** as faithfulness metrics measure a "drop in probability". Indeed, as it is better to look at attributions for models that "dropped" the final softmax layer, it is assumed that it should be added when using metrics object.

```python
def classif_metrics_operator(model: Callable,
                             inputs: tf.Tensor,
                             targets: tf.Tensor) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples. However, this time
    softmax or sigmoid are needed to correctly compute metrics this time while it was remove to
    compute attributions values so we add it here.

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
        Probability scores computed, only for the label class.
    """
    scores = tf.reduce_sum(tf.nn.softmax(model(inputs)) * targets, axis=-1)
    return scores
```

!!!warning
    For classification tasks, you should remove the final softmax layer of your model if you did not do it when computing attribution scores as a softmax will be apply after the call of the model on the inputs!

### Existing operators and how to use them

At present, there are at present 4 (+1) operators available in the library:

- The `predictions_operator` (name='CLASSIFICATION') which is the default operator and the one designed for classification tasks
- (The `classif_metrics_operator` (name='CLASSIFICATION') which is the operator for classification tasks for metrics object)
- The `regression_operator` (name='REGRESSION') which compute the the mean absolute error between model's prediction and the target. Target should be the model prediction on non-perturbed input. This operator can be used to compute attributions for all outputs of a regression model.
- The `binary_segmentation_operator` (name='BINARY_SEGMENTATION') which is an operator thought for binary segmentation tasks with images. **More details are to come**
- The `segmentation_operator` (name='SEGMENTATION') which is an operator thought for segmentation tasks with images. **More details are to come**

You can build attribution methods with those operator in two ways:

- Explicitly importing them

```python
from xplique.attributions import Saliency
from xplique.metrics import Deletion
from xplique.commons.operators import binary_segmentation_operator

explainer = Saliency(model, operator=binary_segmentation_operator)
explanations = explainer(inputs, targets)
```

- Use their name

```python
from xplique.attributions import Saliency
from xplique.metrics import Deletion

explainer = Saliency(model, operator='BINARY_SEGMENTATION')
explanations = explainer(inputs, targets)
```

## Examples of applications

**WIP**

