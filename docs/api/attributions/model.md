# Model expectations: What should be the model provided ?

Even though we tried to cover a wide-range of models for the XAI methods to work we based our frameworks on some assumptions which we propose to see here.

## The inputs have expected shape

As a reminder any attribution methods are instanciated with at least three parameters:

- `model`: the model from which we want to obtain attributions (e.g: InceptionV3, ResNet, ...)
- `batch_size`: an integer which allows to either process inputs per batch (gradient-based methods) or process perturbed samples of an input per batch (inputs are therefore process one by one)
- `operator`: function g to explain, see the [Operator documentation](../operator/) for more details

And an explainer is called with the `explain` method that takes as parameters:

- `inputs`: One of the following: a `tf.data.Dataset` (in which case you should not provide `targets`), a `tf.Tensor` or a `np.ndarray`

- `targets`: One of the following: a `tf.Tensor` or a `np.ndarray`

!!!info
    `targets`' format is defined depending on the task you are doing. See the [Tasks section](#tasks).


### General

In practice, we expect the `model` to be callable for the `inputs` parameters -- *i.e.* we can do `model(inputs)`. We expect this call to produce the `outputs` variables that is the predictions of the model on those inputs. As for most attribution methods we need to manipulate and/or link the `outputs` to the `inputs`  we assume that the latter have conventional shape described in the sections below.

### Inputs Format

#### Images data

If inputs are images, the expected shape of `inputs` is $(N, H, W, C)$ following the TF's conventions where:

- $N$ is the number of inputs
- $H$ is the height of the images
- $W$ is the width of the images
- $C$ is the number of channels (works for $C=3$ or $C=1$, other values might not work or need further customization)

In the case where `inputs` is a `tf.data.Dataset` with images, then we expect each sample of the dataset to be a tuple `(image, target)` with `image` having $(H, W, C)$ shape and target being defined as explained in the [Tasks section](#tasks)

!!!warning
    If your model is not following the same conventions it might lead to poor results or yield errors.

### Tabular data

If inputs are tabular data, the expected shape of `inputs` is $(N, W)$ where:

- $N$ is the number of inputs
- $W$ is the feature dimension of a single input

In the case where `inputs` is a `tf.data.Dataset` with tabular data, then we expect each sample of the dataset to be a tuple `(features, target)` with `features` having $W$ shape and target being a one-hot encoding of the output you want an explanation of.

!!!info
    All attribution methods does not work well with tabular data.

!!!tip
    Please refer to the [table](../../../#whats-included) to see which methods might work with Tabular Data

### Time-Series data

If inputs are Time Series, the expected shape of `inputs` is $(N, T, W)$

- $N$ is the number of inputs
- $T$ is the temporal dimension of a single input
- $W$ is the feature dimension of a single input

!!!warning
    By default `Lime` & `KernelShap` will treat such inputs as grey images. You will need to define a custom `map_to_interpret_space` function when instantiating those methods in order to create a meaningful mapping of Time-Series data into an interpretable space when building such explainers. Such an example is provided at the end of the [Lime's documentation](../lime/).

## Tasks

### Classification Tasks

!!!tip
    In general, if you are doing classification tasks it is better to not include the final softmax layer in your model but to work with logits instead!

For classification tasks, it is expected for the user to use the `predictions_operator` (see the [Operator documentation](../operator/)) when initializing an explainer:

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

#### `model(inputs)`
Consequently, we expect `model(inputs)` to yield a $(N, C)$ tensor or array where $N$ is the number of input samples and $C$ is the number of classes. 

#### `targets`
If you use the default operator for classification task we expect `targets` to be a $(N, C)$ tensor or array which is a one-hot encoding of **the class you want to explain** where $N$ is the number of input samples and $C$ is the number of classes.

### Regression Tasks

If the task at end is regression, then the user should instantiate the explainer with the `regression_operator` (see the [Operator documentation](../operator/)):

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

#### `model(inputs)`
Consequently, we expect `model(inputs)` to yield a $(N, D)$ tensor or array where $N$ is the number of input samples and $D$ is the number of variables the model should predict (possibly one). 

#### `targets`
If you are using the `regression_operator`, it is expected that `targets` to be a $(N, D)$ tensor or array of the expected multi-variate output where $N$ is the number of input samples and $D$ is the number of variables (possibly one).

### Object-Detection Tasks

**Work In Progress**

### Segmentation Tasks

**Work In Progress**

## What if my inputs and/or targets and/or my model does not follow the previous assumptions ?

!!!warning
    In any case, when you are out of the scope of the original API, you should take a deep look at the source code to be sure that your Use Case will make sense.

### My inputs follow a different shape convention
In the case where you want to handle images or time series data that does not follow the previous conventions, it is recommended to reshape the data to the expected shape for the explainers (attribution methods) to handle them correctly. Then, you can simply define a wrapper of your model so that data is reshape to your model convenience when it is called.

For example, if you have a `model` that classifies images but want the images to be channel-first (*i.e.* with $(N, C, H, W)$ shape) then you should:

- Move the axis so inputs are $(N, H, W, C)$ for the explainers
- Write the following wrapper for your model:

```python
class ModelWrapper(tf.keras.models.Model):
    def __init__(self, nchw_model):
        super(ModelWrapper, self).__init__()
        self.model = nchw_model

    def __call__(self, nhwc_inputs):
        # transform the NHWC inputs (wanted for the explainers) back to NCHW inputs
        nchw_inputs = self._transform_inputs(nhwc_inputs)
        # make predictions
        outputs = self.nchw_model(nchw_inputs)

        return outputs

    def _transform_inputs(self, nhwc_inputs):
        # include in this function all transformation
        # needed for your model to work with NHWC inputs
        # , here for example we moveaxis from channels last
        # to channels first
        nchw_inputs = np.moveaxis(nhwc_inputs, [3, 1, 2], [1, 2, 3])

        return nchw_inputs

wrapped_model = ModelWrapper(model)
explainer = Saliency(wrapped_model)
# images should be (N, H, W, C) for the explain call
explanations = explainer.explain(images, labels)
```

### My inputs are a dictionnary (ex: Attention Model)

**Work In Progress**

### My model is neither for classification nor regression tasks

Then you could define your own operator! (see the [Operator documentation](../operator/))
 
### I have a PyTorch model

Then you should definetely have a look on the [dedicated documentation](../../../pytorch/)!

### I have a model that is neither a tf.keras.Model nor a torch.nn.Module

Then you should take a look on the [Callable documentation](../../../callable/) or you could take inspiration on the [PyTorch Wrapper](../../../pytorch/) to write a wrapper that will integrate your model into our API!
