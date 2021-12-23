## 📞 Callable or Models handle by BlackBox Attribution methods

The model can be something else than a `tf.keras.Model` if it respects one of the following condition:
- `model(inputs: np.ndarray)` return either a `np.ndarray` or a `tf.Tensor` of shape $(N, L)$ where $N$ is the number of samples and $L$ the number of targets
- The model has a `scikit-learn` API and has a `predict_proba` function
- The model is a `xgboost.XGBModel` from the [XGBoost python library](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
- The model is a [TF Lite model](https://www.tensorflow.org/api_docs/python/tf/lite). Note this feature is experimental.

On the other hand, a PyTorch model can be used with method having Callable as type of model. In order to makes it work you should write a
wrapper as follow:

```python
class TemplateTorchWrapper(nn.Module):
  def __init__(self, torch_model):
    super(TemplateTorchWrapper, self).__init__()
    self.model = torch_model

  def __call__(self, inputs):
    # transform your numpy inputs to torch
    torch_inputs = self._transform_np_inputs(inputs)
    # mak predictions
    with torch.no_grad():
        outputs = self.model(torch_inputs)
    # convert to numpy
    outputs = outputs.detach().numpy()
    # convert to tf.Tensor
    outputs = tf.cast(outputs, tf.float32)
    return outputs

  def _transform_np_inputs(self, np_inputs):
    # include in this function all transformation
    # needed for your torch model to work, here
    # for example we swap from channels last to
    # channels first
    np_inputs = np.swapaxes(np_inputs, -1, 1)
    torch_inputs = torch.Tensor(np_inputs)
    return torch_inputs

wrapped_model = TemplateTorchWrapper(torch_model)
explainer = Lime(wrapped_model)
explanations = explainer.explain(images, labels)
```

As a matter of fact, if the instance of your model doesn't belong to [`tf.keras.Model`, `tf.lite.Interpreter`, `sklearn.base.BaseEstimator`, `xgboost.XGBModel`] when the explainer will need
to make inference the following will happen:

```python
# inputs are automatically transform to tf.Tensor when using an explainer
pred = model(inputs.numpy())
pred = tf.cast(pred, dtype=tf.float32)
scores = tf.reduce_sum(pred * targets, axis=-1)
```
Knowing that, you are free to wrap your model to make it work with our API!
