# API: Attributions Methods

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>



## Context
In 2013, [Simonyan et al.](http://arxiv.org/abs/1312.6034) proposed a first attribution method, opening the way to a wide range of approaches which could be defined as follow:

!!!definition
    The main objective in attributions techniques is to highlight the discriminating variables for decision-making. For instance, with Computer Vision (CV) tasks, the main goal is to underline the pixels contributing the most in the input image(s) leading to the model’s output(s).

## Common API

All attribution methods inherit from the Base class `BlackBoxExplainer`. This base class can be initialized with two parameters:

- `model`: the model from which we want to obtain attributions (e.g: InceptionV3, ResNet, ...)
- `batch_size`: an integer which allows to either process inputs per batch (gradient-based methods) or process perturbed samples of an input per batch (inputs are therefore process one by one)

In addition, all class inheriting from `BlackBoxExplainer` should implement an `explain` method:

```python
@abstractmethod
def explain(self,
            inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
            targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
    raise NotImplementedError()

def __call__(self,
             inputs: tf.Tensor,
             labels: tf.Tensor) -> tf.Tensor:
    """Explain alias"""
    return self.explain(inputs, labels)
```

`inputs`: Must be one of the following: a `tf.data.Dataset` (in which case you should not provide targets), a `tf.Tensor` or a `np.ndarray`.

- \- If inputs are images, the expected shape of `inputs` is $(N, W, H, C)$ following the TF's conventions where:
    - \- $N$ is the number of inputs
    - \- $W$ is the width of the images
    - \- $H$ is the height of the images
    - \- $C$ is the number of channels (works for $C=3$ or $C=1$, other values might not work or need further customization)

- \- If inputs are tabular data, the expected shape of `inputs` is $(N, W)$ where:
    - \- $N$ is the number of inputs
    - \- $W$ is the feature dimension of a single input

    !!!tip
        Please refer to the [table](https://deel-ai.github.io/xplique/#whats-included) to see which methods might work with Tabular Data

- \- (Experimental) If inputs are Time Series, the expected shape of `inputs` is $(N, T, W)$
    - \- $N$ is the number of inputs
    - \- $T$ is the temporal dimension of a single input
    - \- $W$ is the feature dimension of a single input

        !!!warning
            By default `Lime` & `KernelShap` will treat such inputs as grey images. You will need to define a custom `map_to_interpret_space` when building such interpreters.

    !!!warning
        If your model is not following the same conventions it might lead to poor results.

    On the bright side, there is only need for your `model` to be called on `inputs` with such shape. Therefore, you can overcome this by writing a wrapper around your model.
    
    <br/>
    For example, imagine you have a trained model which takes images with channel first (*i.e* $inputs.shape=(N, C, W, H)$). However, we saw that an explainer need images inputs
    with $(N, W, H, C)$ shape. Then, we can wrap the original model and redefine its call function so that it swaps inputs axes before proceding to the original call:

```python
class TemplateModelWrapper(nn.Module):
    def __init__(self, ncwh_model):
        super(TemplateModelWrapper, self).__init__()
        self.model = ncwh_model

    def __call__(self, nwhc_inputs):
        # transform your NWHC inputs to NCWH inputs
        nchw_inputs = self._transform_inputs(nwhc_inputs)
        # make predictions
        outputs = self.ncwh_model(nchw_inputs)

        return outputs

    def _transform_inputs(self, nwhc_inputs):
        # include in this function all transformation
        # needed for your model to work with NWHC inputs
        # , here for example we swap from channels last
        # to channels first
        ncwh_inputs = tf.transpose(nwhc_inputs, [0, 3, 1, 2])

        return ncwh_inputs

wrapped_model = TemplateModelWrapper(model)
explainer = Saliency(wrapped_model)
# images should be (N, W, H, C) for the explain call
explanations = explainer.explain(images, labels)
```

!!!warning
    In any case, when you are out of the scope of the original API, you should take a deep look at the source code to be sure that your Use Case will make sense.

`targets`: Must be one of the following: a `tf.Tensor` or a `np.ndarray`.

!!!info
    `targets` should be a one hot encoding of the output you want an explanation of!

-   \- Therefore, targets's shape must match: $(N, outputs\_size)$
    - \- $N$ is the number of inputs
    - \- $outputs\_size$ is the number of outputs
-   \- For a classification task, the $1$ value should be on the class of interest's (and only this one) index on the outputs. For example, I have three classes ('dogs, 'cats', 'fish') and a classifier with three outputs (the probability to belong to each class). I have an image of a fish and I want to know why my model think it is a fish. Then, the corresponding target of my image will be $[0, 0, 1]$

    !!!warning
        Sometimes the explanation might be non-sense. One possible reason is that your model did not predict
        at all the output you asked an explanation for. For example, in the previous configuration, the model might have predicted a cat on your fish image. Therefore, you might want to see why it made such a prediction and use $[0, 1, 0]$ as target.
        
    !!!tip
        If you replace $1$ by $-1$ you can also see what goes against an output prediction!
-   \- For a regression task, you might have only one output then one target will be the vector $[1]$ (and not the regression value!)

Even though we made an harmonized API for all attributions methods it might be relevant for the user to distinguish Gradient based and Perturbation based methods, also often referenced respectively as white-box and black-box methods, as their hyperparameters settings might be quite different.

## Perturbation-based approaches

Perturbation based methods focus on perturbing an input with a variety of techniques and, with the analysis of the resulting outputs, define an attribution representation. Therefore, **there is no need to explicitly know the model architecture** as long as forward pass is available, which explain why they are also referenced as black-box methods.

Therefore, to use perturbation-based approaches you do not need a TF model. To know more, please see the [Callable](https://deel-ai.github.io/xplique/callable.html) documentation.

Xplique includes the following black-box attributions:

| Method Name      | **Tutorial**             |
|:---------------- | :----------------------: |
| KernelShap              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT?authuser=1) |
| Lime                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT?authuser=1) |
| Occlusion               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q?authuser=1) |
| Rise                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO?authuser=1) |

## Gradient-based approaches

Those approaches are also called white-box methods as **they require a full access to the model architecture**, notably it should **allow computing gradients with TensorFlow** (for Xplique, in general any automatic differentiation framework would work). Indeed, the core idea with the gradient-based approach is to use back-propagation, along other techniques, not to update the model’s weights (which is already trained) but to reveal the most contributing inputs, potentially in a specific layer. Xplique includes the following white-box attributions:

| Method Name      | **Tutorial**             |
|:---------------- | :----------------------: |
| DeconvNet               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) |
| GradCAM                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X?authuser=1) |
| GradCAM++               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X?authuser=1) |
| GradientInput           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) |
| GuidedBackpropagation   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) |
| IntegratedGradients     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo?authuser=1) |
| Saliency                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) |
| SmoothGrad              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD?authuser=1) |
| SquareGrad              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD?authuser=1) |
| VarGrad                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD?authuser=1) |

In addition, those methods inherits from `WhiteBoxExplainer` (itself inheriting from `BlackBoxExplainer`). Thus, an additional `__init__` argument is added: `output_layer`. It is the layer to target for the output (e.g logits or after softmax). If an `int` is provided, it will be interpreted as a layer index, if a `string` is provided it will look for the layer name. Default to the last layer.

!!!tip
    It is recommended to use the layer before Softmax