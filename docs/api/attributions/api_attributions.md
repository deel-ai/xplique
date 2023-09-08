# API: Attributions Methods

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>







## Context ##

In 2013, [Simonyan et al.](http://arxiv.org/abs/1312.6034) proposed a first attribution method, opening the way to a wide range of approaches which could be defined as follow:

!!!definition
    The main objective in attributions techniques is to highlight the discriminating variables for decision-making. For instance, with Computer Vision (CV) tasks, the main goal is to underline the pixels contributing the most in the input image(s) leading to the model’s output(s).







## Common API ##

```python
explainer = Method(model, batch_size, operator)
explanation = explainer(inputs, targets)
```

The API have two steps:

- **`explainer` instantiation**: `Method` is an attribution method among those displayed [methods tables](#methods). It inherits from the Base class `BlackBoxExplainer`. Their initialization takes 3 parameters apart from the specific ones and generates an `explainer`:
    - `model`: the model from which we want to obtain attributions (e.g: InceptionV3, ResNet, ...), see the [model section](#model) for more details and specifications.
    - `batch_size`: an integer which allows to either process inputs per batch (gradient-based methods) or process perturbed samples of an input per batch (inputs are therefore processed one by one).
    - `operator`: enum identifying the task of the model (which is [Classification](../classification/) by default), string identifying the task, or function to explain, see the [task and operator section](#tasks-and-operator) for more detail.

- **`explainer` call**: The call to `explainer` generates the explanations, it takes two parameters:
    - `inputs`: the samples on which the explanations are requested, see [inputs section](#inputs) for more detail.
    - `targets`: another parameter to specify what to explain in the `inputs`, it changes depending on the `operator`, see [targets section](#targets) for more detail.

!!!info
    The `__call__` method of explainers is an alias for the `explain` method.

!!!info
    This documentation page covers the different parameters of the common API of attributions methods. It is common between the different [tasks covered](#the-tasks-covered) by Xplique for attribution methods.







## Methods ##


Even though we made an harmonized API for all attributions methods, it might be relevant for the user to distinguish [Perturbation-based methods](#perturbation-based-approaches) and [Gradient-based methods](#gradient-based-approaches), also often referenced respectively as black-box and white-box methods, as their hyperparameters settings might be quite different.



### Perturbation-based approaches ###

Perturbation based methods focus on perturbing an input with a variety of techniques and, with the analysis of the resulting outputs, define an attribution representation. Thus, **there is no need to explicitly know the model architecture** as long as forward pass is available, which explains why they are also referenced as black-box methods.

Therefore, to use perturbation-based approaches you do not need a TF model. To know more, please see the [Callable](../callable/) documentation.

Xplique includes the following black-box attributions:

| Method Name and Documentation link     | **Tutorial**             | Available with TF | Available with PyTorch* |
|:-------------------------------------- | :----------------------: | :---------------: | :---------------------: |
| [KernelShap](../methods/kernel_shap/)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) | ✔ | ✔ |
| [Lime](../methods/lime/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) | ✔ | ✔ |
| [Occlusion](../methods/occlusion/)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q) | ✔ | ✔ |
| [Rise](../methods/rise/)               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO) | ✔ | ✔ |
| [Sobol Attribution](../methods/sobol/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) | ✔ | ✔ |
| [Hsic Attribution](../methods/hsic/)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) | ✔ | ✔ |

*: Before using a PyTorch model it is highly recommended to read the [dedicated documentation](../pytorch/)



### Gradient-based approaches ###

Those approaches are also called white-box methods as **they require a full access to the model's architecture**, notably it must **allow computing gradients**. Indeed, the core idea with the gradient-based approaches is to use back-propagation, not to update the model’s weights (which is already trained) but to reveal the most contributing inputs, potentially in a specific layer. All methods are available when the model works with TensorFlow but most methods also work with PyTorch (see [Xplique for PyTorch documentation](../pytorch/))

| Method Name and Documentation link                          | **Tutorial**             | Available with TF | Available with PyTorch* |
|:----------------------------------------------------------- | :----------------------: | :---------------: | :---------------------: |
| [DeconvNet](../methods/deconvnet/)                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ |❌ |
| [GradCAM](../methods/grad_cam/)                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) | ✔ |❌ |
| [GradCAM++](../methods/grad_cam_pp/)                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) | ✔ |❌ |
| [GradientInput](../methods/gradient_input/)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ | ✔ |
| [GuidedBackpropagation](../methods/guided_backpropagation/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ |❌ |
| [IntegratedGradients](../methods/integrated_gradients/)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo) | ✔ | ✔ |
| [Saliency](../methods/saliency/)                            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ | ✔ |
| [SmoothGrad](../methods/smoothgrad/)                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |
| [SquareGrad](../methods/square_grad/)                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |
| [VarGrad](../methods/vargrad/)                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |

*: Before using a PyTorch model it is highly recommended to read the [dedicated documentation](../pytorch/)

In addition, these methods inherit from `WhiteBoxExplainer` (itself inheriting from `BlackBoxExplainer`). Thus, an additional `__init__` argument is added: `output_layer`. It is the layer to target for the output (e.g logits or after softmax). If an `int` is provided, it will be interpreted as a layer index, if a `string` is provided it will look for the layer name. Default to the last layer.

!!!tip
    It is recommended to use the layer before Softmax.

!!!warning
    The `output_layer` parameter will work well with TensorFlow models. However, it will not work with PyTorch models. For PyTorch, one should directly manipulate the model to focus on the layers of interest.

!!!info
    The "white-box" explainers that work with PyTorch are those that only require the gradient of the model without having to "modify" some part of the model (e.g. Deconvnet will commute all original ReLU by a custom ReLU policy)







## `model` ##

`model` is the primary parameter of attribution methods: it represents model from which explanations are required. Even though we tried to support a wide-range of models, our attributions framework relies on some assumptions which we propose to see in this section.

!!!warning
    In case the `model` does not respect the specifications, a wrapper will be needed as described in the [Models not respecting the specifications section](#models-not-respecting-the-specifications).

In practice, we expect the `model` to be callable for the `inputs` parameters -- *i.e.* we can do `model(inputs)`. We expect this call to produce the `outputs` variables that are the predictions of the model on those inputs. As for most attribution methods, we need to manipulate and/or link the `outputs` to the `inputs`. We assume that the latter follow conventional shapes described in the [inputs section](#inputs).

!!!info
    Depending on the [task and operator](#tasks-and-operator) there may be supplementary specifications for the model, mainly on the output of the model.







## Tasks and `operator` ##

`operator` is one of the main parameters for both attribution methods and [metrics](../metrics/api_metrics/). It defines the function that we want to explain. *E.g.*: In the case we have a classifier model, the function that we might want to explain is the one that given a target provides us the score of the model for that specific target -- *i.e* $model(input)[target]$.

!!!note
    The `operator` parameter is a feature available for version > $1.$. The `operator` default values are the ones used before the introduction of this new feature!



### Leitmotiv ###

The `operator` parameter was introduced to offer users a flexible way to adapt current attribution methods or metrics. It should help them to empirically tackle new use-cases/new tasks. Broadly speaking, it should amplify the user's ability to experiment. However, this also implies that it is the user's responsibility to make sure that its derivations are in-scope of the original method and make sense.



### `operator` in practice ###

In practice, the user does not manipulate the function in itself. The use of the operator can be divided in three steps:

- Specify the operator to use in the method initialization (as shown in the [API description](#api-attributions-methods)). Possible values are either an enum encoding the [task](#the-tasks-covered), a string, or a [custom operator](#providing-custom-operator).
- Make sure the model follows the model's specification relative to the selected [task](#the-tasks-covered).
- Specify what to explain in `inputs` through `targets`, the `targets` parameter specifications depend on the [task](#the-tasks-covered).



### The tasks covered ###

The `operator` parameter depends on the task to explain, as the function to explain depends on the task. In the case of Xplique, the tasks in the following table are supported natively, but new operators are welcome, please feel free to contribute.

| Task and Documentation link                        | `operator` parameter value <br/> from `xplique.Tasks` Enum  | Tutorial link |
| :------------------------------------------------- | :---------------------------------------------------------- | :------------ |
| [Classification](../classification/)               | `CLASSIFICATION`        | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub> |
| [Object Detection](../object_detection/)           | `OBJECT_DETECTION`      | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub> |
| [Regression](../regression/)                       | `REGRESSION`            | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub> |
| [Semantic Segmentation](../semantic_segmentation/) | `SEMANTIC_SEGMENTATION` | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) </sub> |

!!!info
    Classification is the default behavior, *i.e.*, if no `operator` value is specified or `None` is given.

!!!warning
    To apply Xplique on different tasks, specifying the value of the `operator` is not enough. Be sure to respect the ["operator in practice" steps](#operator-in-practice).



### Operators' Signature ###

An `operator` is a function that we want to explain. This function takes as input $3$ parameters:

- the `model` to explain as in the method instantiation (specifications in the [model section](#model)).
- the `inputs` parameter representing the samples to explain as in method call (specifications in [inputs section](#inputs)).
- the `targets` parameter encoding what to explain in the `inputs` (specifications in [targets section](#targets)).

This function should return a **vector of scalar value** of size $(N,)$ where $N$ is the number of inputs in `inputs` -- *i.e* a scalar score per input.

!!!note
    For [gradient-based methods](#gradient-based-approaches) to work with the `operator`, it needs to be differentiable with respect to `inputs`.



### The operators mechanism ###

??? info "Operators behavior for Black-box attribution methods"

    For attribution approaches that do not require gradient computation, we mostly need to query the model. Thus, those methods need an inference function. If you provide an `operator`, it will be the inference function.

    More concretely, for this kind of approach, you want to compare some valued function for an original input and perturbed version of it:

    ```python
    original_scores = operator(model, original_inputs, original_targets)

    # depending on the attribution method, this `perturbation_function` is different
    perturbed_inputs, perturbed_targets = perturbation_function(original_inputs, original_targets)
    perturbed_scores = operator(model, perturbed_inputs, perturbed_targets)

    # example of comparison of interest
    diff_scores = math.sqrt((original_scores - perturbed_scores)**2)
    ```

??? info "Operators behavior for White-box attribution methods"

    These methods usually require some gradients computation. The gradients that will be used are the ones of the operator function (see the `get_gradient_of_operator` method in the [Providing custom operator](#providing-custom-operator) section).



### Providing custom operator ###

The `operator` parameter also supports functions (*i.e.* `Callable`), this is considered a custom operator and in this case, you should be aware of the following points:

- An assertion will be made to ensure it respects [operators' signature](#operators-signature).
- If you use any white-box explainer, your operator will go through the `get_gradient_of_operator` function below.

??? example "Code of the `get_gradient_of_operator` function."

    ```python
    def get_gradient_of_operator(operator):
        """
        Get the gradient of an operator.

        Parameters
        ----------
        operator
            Operator of which to compute the gradient.

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
    Writing your operator with only tensorflow functions should increase your chance that this method does not yield any errors. In addition, providing a `@tf.function` decorator is also welcome!

!!!warning
    The `targets` parameter is the key to specifying what to explain and differs greatly depending on the operator. 







## Models not respecting the specifications ##

!!!warning
    In any case, when you are out of the scope of the original API, you should take a deep look at the source code to be sure that your Use Case will make sense.



### My inputs follow a different shape convention

In the case where you want to handle images or time series data that does not follow the previous conventions, it is recommended to reshape the data to the expected shape for the explainers (attribution methods) to handle them correctly. Then, you can simply define a wrapper of your model so that data is reshape to your model convenience when it is called.

For example, if you have a `model` that classifies images but want the images to be channel-first (*i.e.* with $(N, C, H, W)$ shape) then you should:

- Move the axis so inputs are $(N, H, W, C)$ for the explainers
- Write the following wrapper for your model:

??? example "Example of a wrapper."

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
            # , here for example we move axis from channels last
            # to channels first
            nchw_inputs = np.moveaxis(nhwc_inputs, [3, 1, 2], [1, 2, 3])

            return nchw_inputs

    wrapped_model = ModelWrapper(model)
    explainer = Saliency(wrapped_model)
    # images should be (N, H, W, C) for the explain call
    explanations = explainer.explain(images, labels)
    ```
 
### I have a PyTorch model

Then you should definitely take a look at the [PyTorch documentation](../pytorch/)!

### I have a model that is neither a tf.keras.Model nor a torch.nn.Module

Then you should take a look at the [Callable documentation](../callable/) or you could take inspiration on the [PyTorch Wrapper](../pytorch/) to write a wrapper that will integrate your model into our API!







## `inputs` ##

!!!Warning
    `inputs` in this section correspond to the argument in the `explain` method of `BlackBoxExplainer`. The `model` specified at the initialization of the `BlackBoxExplainer` should be able to be called through `model(inputs)`. Otherwise, a wrapper needs to be implemented as described in the [Models not respecting the specifications section](#models-not-respecting-the-specifications).

`inputs`: Must be one of the following: a `tf.data.Dataset` (in which case you should not provide targets), a `tf.Tensor` or a `np.ndarray`.

- If inputs are images, the expected shape of `inputs` is $(N, H, W, C)$ following the TF's conventions where:
    - $N$: the number of inputs
    - $H$: the height of the images
    - $W$: the width of the images
    - $C$: the number of channels (works for $C=3$ or $C=1$, other values might not work or need further customization)

- If inputs are tabular data, the expected shape of `inputs` is $(N, W)$ where:
    - $N$: the number of inputs
    - $W$: the feature dimension of a single input

    !!!tip
        Please refer to the [table of attributions available](../../../#whats-included) to see which methods might work with Tabular Data.

- (Experimental) If inputs are Time Series, the expected shape of `inputs` is $(N, T, W)$
    - $N$: the number of inputs
    - $T$: the temporal dimension of a single input
    - $W$: the feature dimension of a single input

        !!!warning
            By default `Lime` & `KernelShap` will treat such inputs as grey images. You will need to define a custom `map_to_interpret_space` function when instantiating these methods in order to create a meaningful mapping of Time-Series data into an interpretable space when building such explainers. An example of this is provided at the end of the [Lime's documentation](../methods/lime/).

    !!!note
        If your model is not following the same conventions, please refer to the [model not respecting the specification documentation](#models-not-respecting-the-specifications).







## `targets` ##

`targets`: Must be one of the following: a `tf.Tensor` or a `np.ndarray`. It has a shape of $(N, ...)$ where N should match the first dimension of `inputs`, while $...$ depend on the task and operators. Indeed, the `targets` parameter is highly dependent on the `operator` selected for the attribution methods, hence, for more information please refer to the [tasks and operators table](#tasks-and-operator) which will lead you to the pertinent task documentation page.
