# API: Example-based API

- [**Example-based Methods**: Getting strated]() **WIP**

## Context ##

!!! quote
    While saliency maps have stolen the show for the last few years in the XAI field, their ability to reflect models' internal processes has been questioned. Although less in the spotlight, example-based XAI methods have continued to improve. It encompasses methods that use examples as explanations for a machine learning model's predictions. This aligns with the psychological mechanisms of human reasoning and makes example-based explanations natural and intuitive for users to understand. Indeed, humans learn and reason by forming mental representations of concepts based on examples.

    -- <cite>[Natural Example-Based Explainability: a Survey (2023)](https://arxiv.org/abs/2309.03234)</cite>[^1]

As mentioned by our team members in the quote above, example-based methods are an alternative to saliency maps and can be more aligned with some users' expectations. Thus, we have been working on implementing some of those methods in Xplique that have been put aside in the previous developments.

While not being exhaustive we tried to cover a range of methods that are representative of the field and that belong to different families: similar examples, contrastive (counter-factuals and semi-factuals) examples, and prototypes (as concepts based methods have a dedicated sections).

At present, we made the following choices:
- Focus on methods that are natural example methods (see the paper above for more details).
- Try to unify the three families of approaches with a common API.

!!! info
    We are in the early stages of development and are looking for feedback on the API design and the methods we have chosen to implement. Also, we are counting on the community to furnish the collection of methods available. If you are willing to contribute reach us on the [GitHub](https://github.com/deel-ai/xplique) repository (with an issue, pull request, ...).

## Common API ##

```python
explainer = ExampleMethod(
    cases_dataset,
    labels_dataset,
    targets_dataset,
    k,
    projection,
    case_returns,
    batch_size,
    **kwargs
)

explanations = explainer.explain(inputs, targets)
```

We tried to keep the API as close as possible to the one of the attribution methods to keep a consistent experience for the users.

The `BaseExampleMethod` is an abstract base class designed for example-based methods used to explain classification models. It provides examples from a dataset (usually the training dataset) to help understand a model's predictions. Examples are selected using a [search method](#search-methods) within a defined search space, projected from the input space using a [projection function](#projections).

??? abstract "Table of example-based methods available"

    | Method | Documentation | Family |
    | --- | --- | --- |
    | `SimilarExamples` | [SimilarExamples](api/example_based/methods/similar_examples) | Similar Examples |
    | `Cole` | [Cole](api/example_based/methods/cole) | Similar Examples |
    | `ProtoGreedy` | [ProtoGreedy](api/example_based/methods/proto_greedy/) | Prototypes |
    | `ProtoDash` | [ProtoDash](api/example_based/methods/proto_dash/) | Prototypes |
    | `MMDCritic` | [MMDCritic](api/example_based/methods/mmd_critic/) | Prototypes |
    | `NaiveCounterFactuals` | [NaiveCounterFactuals](api/example_based/methods/naive_counter_factuals/) | Counter Factuals |
    | `LabelAwareCounterFactuals` | [LabelAwareCounterFactuals](api/example_based/methods/label_aware_counter_factuals/) | Counter Factuals |
    | `KLEORSimMiss` | [KLEOR](api/example_based/methods/kleor/) | Semi Factuals |
    | `KLEORGlobalSim` | [KLEOR](api/example_based/methods/kleor/) | Semi Factuals |

### Parameters ###

- **cases_dataset** (`Union[tf.data.Dataset, tf.Tensor, np.ndarray]`): The dataset used to train the model, from which examples are extracted. It should be batched as TensorFlow provides no method to verify this. Ensure the dataset is not reshuffled at each iteration.
- **labels_dataset** (`Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]`): Labels associated with the examples in the cases dataset. Indices should match the `cases_dataset`.
- **targets_dataset** (`Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]`): Targets associated with the `cases_dataset` for dataset projection, often the one-hot encoding of a model's predictions.
- **k** (`int`): The number of examples to retrieve per input.
- **projection** (`Union[Projection, Callable]`): A projection or callable function that projects samples from the input space to the search space. The search space should be relevant for the model. (see [Projections](#projections))
- **case_returns** (`Union[List[str], str]`): Elements to return in `self.explain()`. Default is "examples".
- **batch_size** (`Optional[int]`): Number of samples processed simultaneously for projection and search. Ignored if `tf.data.Dataset` is provided.

### Properties ###

- **search_method_class** (`Type[BaseSearchMethod]`): Abstract property to define the search method class to use. Must be implemented in subclasses. (see [Search Methods](#search-methods))
- **k** (`int`): Getter and setter for the `k` parameter.
- **returns** (`Union[List[str], str]`): Getter and setter for the `returns` parameter. Defines the elements to return in `self.explain()`.

### `explain(self, inputs, targets)` ###

Returns the relevant examples to explain the (inputs, targets). Projects inputs using `self.projection` and finds examples using the `self.search_method`.

- **inputs** (`Union[tf.Tensor, np.ndarray]`): Input samples to be explained.
- **targets** (`Optional[Union[tf.Tensor, np.ndarray]]`): Targets associated with the `cases_dataset` for dataset projection.

**Returns:** Dictionary with elements listed in `self.returns`.

!!!info
    The `__call__` method is an alias for the `explain` method.

## Projections ##
Projections are functions that map input samples to a search space where examples are retrieved with a `search_method`. The search space should be relevant for the model (e.g. projecting the inputs into the latent space of the model).

!!!info
    If one decides to use the identity function as a projection, the search space will be the input space, thus rather explaining the dataset than the model. In this case, it may be more relevant to directly use a `search_method` ([Search Methods](#search-methods)) for the dataset.

The `Projection` class is an abstract base class for projections. It involves two parts: `space_projection` and `weights`. The samples are first projected to a new space and then weighted. 

!!!warning
    If both parts are `None`, the projection acts as an identity function. At least one part should involve the model to ensure meaningful distance calculations.

??? abstract "Table of projection methods available"

    | Method | Documentation |
    | --- | --- |
    | `Projection` | HERE |
    | `LatentSpaceProjection`| [LatentSpaceProjection](api/example_based/projections/latent_space_projection/) |
    | `HadamardProjection` | [HadamardProjection](api/example_based/projections/hadamard_projection/) |
    | `AttributionProjection` | [AttributionProjection](api/example_based/projections/attribution_projection/) |

### Parameters ###

- **get_weights** (`Optional[Union[Callable, tf.Tensor, np.ndarray]]`): Either a Tensor or a callable function. 
  - **Tensor**: Weights are applied in the projected space.
  - **Callable**: A function that takes inputs and targets, returning the weights (Tensor). Weights should match the input shape (possibly differing in channels).
  
  **Example**:
  ```python
    def get_weights_example(projected_inputs: Union[tf.Tensor, np.ndarray],
                            targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        # Compute weights using projected_inputs and targets.
        weights = ...  # Custom logic involving the model.
        return weights
  ```

- **space_projection** (`Optional[Callable]`): Callable that takes samples and returns a Tensor in the projected space. An example of a projected space is the latent space of a model.
- **device** (`Optional[str]`): Device to use for the projection. If `None`, the default device is used.
- **mappable** (`bool`): If `True`, the projection can be applied to a dataset through `Dataset.map`. Otherwise, the projection is done through a loop.

### `project(self, inputs, targets=None)` ###

Projects samples into a space meaningful for the model. This involves weighting the inputs, projecting them into a latent space, or both. This method should be called during initialization and for each explanation.

- **inputs** (`Union[tf.Tensor, np.ndarray]`): Input samples to be explained. Expected shapes include (N, W), (N, T, W), (N, W, H, C).
- **targets** (`Optional[Union[tf.Tensor, np.ndarray]]`): Additional parameter for `self.get_weights` function.

**Returns:** `projected_samples` - The samples projected into the new space.

!!!info
    The `__call__` method is an alias for the `project` method.

### `project_dataset(self, cases_dataset, targets_dataset=None)` ###

Applies the projection to a dataset through `Dataset.map`.

- **cases_dataset** (`tf.data.Dataset`): Dataset of samples to be projected.
- **targets_dataset** (`Optional[tf.data.Dataset]`): Dataset of targets for the samples.

**Returns:** `projected_dataset` - The projected dataset.

## Search Methods ##

Search methods are used to retrieve examples from the `cases_dataset` that are relevant to the input samples.

!!!info
    In an Example method, the `cases_dataset` is the dataset that has been projected with a `Projection` object (see the previous section). The search methods are used to find examples in this projected space.

The `BaseSearchMethod` class is an abstract base class for example-based search methods. It defines the interface for search methods used to find examples in a dataset. This class should be inherited by specific search methods.

??? abstract "Table of search methods available"

    | Method | Documentation |
    | --- | --- |
    | `KNN` | [KNN](api/example_based/search_methods/knn/) |
    | `FilterKNN` | [KNN](api/example_based/search_methods/knn/) |
    | `ProtoGreedySearch` | [ProtoGreedySearch](api/example_based/search_methods/proto_greedy_search/) |
    | `ProtoDashSearch` | [ProtoDashSearch](api/example_based/search_methods/proto_dash_search/) |
    | `MMDCriticSearch` | [MMDCriticSearch](api/example_based/search_methods/mmd_critic_search/) |
    | `KLEORSimMissSearch` | [KLEOR](api/example_based/search_methods/kleor/) |
    | `KLEORGlobalSimSearch` | [KLEOR](api/example_based/search_methods/kleor/) |


### Parameters ###

- **cases_dataset** (`Union[tf.data.Dataset, tf.Tensor, np.ndarray]`): The dataset containing the examples to search in. It should be batched as TensorFlow provides no method to verify this. Ensure the dataset is not reshuffled at each iteration.
- **k** (`int`): The number of examples to retrieve.
- **search_returns** (`Optional[Union[List[str], str]]`): Elements to return in `self.find_examples()`. It should be a subset of `self._returns_possibilities`.
- **batch_size** (`Optional[int]`): Number of samples treated simultaneously. It should match the batch size of the `cases_dataset` if it is a `tf.data.Dataset`.

### Properties ###

- **k** (`int`): Getter and setter for the `k` parameter.
- **returns** (`Union[List[str], str]`): Getter and setter for the `returns` parameter. Defines the elements to return in `self.find_examples()`.

### `find_examples(self, inputs, targets)` ###

Abstract method to search for samples to return as examples. It should be implemented in subclasses. It may return the indices corresponding to the samples based on `self.returns` value.

- **inputs** (`Union[tf.Tensor, np.ndarray]`): Input samples to be explained. Expected shapes include (N, W), (N, T, W), (N, W, H, C).
- **targets** (`Optional[Union[tf.Tensor, np.ndarray]]`): Targets associated with the samples to be explained.

**Returns:** `return_dict` - Dictionary containing the elements specified in `self.returns`.

!!!info
    The `__call__` method is an alias for the `find_examples` method.

### `_returns_possibilities`

Attribute thet list possible elements that can be returned by the search methods. For the base class: `["examples", "distances", "labels", "include_inputs"]`.

[^1]: [Natural Example-Based Explainability: a Survey (2023)](https://arxiv.org/abs/2309.03234)