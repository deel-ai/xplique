# API: Example-based

- [**Example-based Methods**: Getting started](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) </sub>
- [**Example-based: Prototypes**](https://colab.research.google.com/drive/1OI3oa884GwGbXlzn3Y9NH-1j4cSaQb0w) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OI3oa884GwGbXlzn3Y9NH-1j4cSaQb0w) </sub>

## Context ##

!!! quote
    While saliency maps have stolen the show for the last few years in the XAI field, their ability to reflect models' internal processes has been questioned. Although less in the spotlight, example-based XAI methods have continued to improve. It encompasses methods that use samples as explanations for a machine learning model's predictions. This aligns with the psychological mechanisms of human reasoning and makes example-based explanations natural and intuitive for users to understand. Indeed, humans learn and reason by forming mental representations of concepts based on examples.

    -- <cite>[Natural Example-Based Explainability: a Survey (2023)](https://arxiv.org/abs/2309.03234)</cite>[^1]

As mentioned by our team members in the quote above, example-based methods are an alternative to saliency maps and can be more aligned with some users' expectations. Thus, we have been working on implementing some of those methods in Xplique that have been put aside in the previous developments.

While not being exhaustive we tried to cover a range of methods that are representative of the field and that belong to different families: similar examples, contrastive (counter-factuals and semi-factuals) examples, and prototypes (as concepts based methods have a dedicated sections).

At present, we made the following choices:
- Focus on methods that are natural example methods (post-hoc and non-generative, see the paper above for more details).
- Try to unify the four families of approaches with a common API.

!!! info
    We are in the early stages of development and are looking for feedback on the API design and the methods we have chosen to implement. Also, we are counting on the community to furnish the collection of methods available. If you are willing to contribute reach us on the [GitHub](https://github.com/deel-ai/xplique) repository (with an issue, pull request, ...).

## Common API ##

```python
projection = ProjectionMethod(model)

explainer = ExampleMethod(
    cases_dataset=cases_dataset, 
    k=k,
    projection=projection,
    case_returns=case_returns,
    distance=distance,
)

outputs_dict = explainer.explain(inputs, targets)
```

We tried to keep the API as close as possible to the one of the attribution methods to keep a consistent experience for the users.

The `BaseExampleMethod` is an abstract base class designed for example-based methods used to explain classification models. It provides examples from a dataset (usually the training dataset) to help understand a model's predictions. Examples are projected from the input space to a search space using a [projection function](#projections). The projection function defines the search space. Then, examples are selected using a [search method](#search-methods) within the search space. For all example-based methods, one can define the `distance` that will be used by the search method. 

We can broadly categorize example-based methods into four families: similar examples, counter-factuals, semi-factuals, and prototypes.

- **Similar Examples**: This method involves finding instances in the dataset that are similar to a given instance. The similarity is often determined based on the feature space, and these examples can help in understanding the model's decision by showing what other data points resemble the instance in question.
- **Counter Factuals**: Counterfactual explanations identify the minimal changes needed to an instance's features to change the model's prediction to a different, specified outcome. They help answer "what-if" scenarios by showing how altering certain aspects of the input would lead to a different decision.
- **Semi Factuals**: Semifactual explanations describe hypothetical situations where most features of an instance remain the same except for one or a few features, without changing the overall outcome. They highlight which features could vary without altering the prediction.
- **Prototypes**: Prototypes are representative examples from the dataset that summarize typical cases within a certain category or cluster. They act as archetypal instances that the model uses to make predictions, providing a reference point for understanding model behavior. Additional documentation can be found in the [Prototypes API documentation](../prototypes/api_prototypes/).

??? abstract "Table of example-based methods available"

    | Method | Family | Documentation | Tutorial |
    | --- | --- | --- | --- |
    | `SimilarExamples` | Similar Examples | [SimilarExamples](../similar_examples/similar_examples/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    | `Cole` | Similar Examples | [Cole](../similar_examples/cole/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    |  |  |  |
    | `NaiveCounterFactuals` | Counter Factuals | [NaiveCounterFactuals](../counterfactuals/naive_counter_factuals/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    | `LabelAwareCounterFactuals` | Counter Factuals | [LabelAwareCounterFactuals](../counterfactuals/label_aware_counter_factuals/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    ||||
    | `KLEORSimMiss` | Semi Factuals | [KLEOR](../semifactuals/kleor/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    | `KLEORGlobalSim` | Semi Factuals | [KLEOR](../semifactuals/kleor/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
    ||||
    | `ProtoGreedy` | Prototypes | [ProtoGreedy](../prototypes/proto_greedy/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OI3oa884GwGbXlzn3Y9NH-1j4cSaQb0w) |
    | `ProtoDash` | Prototypes | [ProtoDash](../prototypes/proto_dash/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OI3oa884GwGbXlzn3Y9NH-1j4cSaQb0w) |
    | `MMDCritic` | Prototypes | [MMDCritic](../prototypes/mmd_critic/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OI3oa884GwGbXlzn3Y9NH-1j4cSaQb0w) |

### Parameters ###

`DatasetOrTensor = Union[tf.Tensor, np.ndarray, "torch.Tensor", tf.data.Dataset, "torch.utils.data.DataLoader"]`

- **cases_dataset** (`DatasetOrTensor`): The dataset used to train the model, examples are extracted from this dataset. All datasets (cases, labels, and targets) should be of the same type. Supported types are: `tf.data.Dataset`, `torch.utils.data.DataLoader`, `tf.Tensor`, `np.ndarray`, `torch.Tensor`. For datasets with multiple columns, the first column is assumed to be the cases. While the second column is assumed to be the labels, and the third the targets. Warning: datasets tend to reshuffle at each iteration, ensure the datasets are not reshuffle as we use index in the dataset.
- **labels_dataset** (`Optional[DatasetOrTensor]`): Labels associated with the examples in the cases dataset. It should have the same type as `cases_dataset`.
- **targets_dataset** (`Optional[DatasetOrTensor]`): Targets associated with the `cases_dataset` for dataset projection, often the one-hot encoding of a model's predictions. See `projection` for detail. It should have the same type as `cases_dataset`. It is not be necessary for all projections. Furthermore, projections which requires it compute it internally by default.
- **k** (`int`): The number of examples to retrieve per input.
- **projection** (`Union[Projection, Callable]`): A projection or callable function that projects samples from the input space to the search space. The search space should be relevant for the model. (see [Projections](#projections))
- **case_returns** (`Union[List[str], str]`): Elements to return in `self.explain()`. Default is `"examples"`. `"all"` indicates that every possible output should be returned.
- **batch_size** (`Optional[int]`): Number of samples processed simultaneously for projection and search. Ignored if `cases_dataset` is a batched `tf.data.Dataset` or a batched `torch.utils.data.DataLoader` is provided.

!!!tips
    If the elements of your dataset are tuples (cases, labels), you can pass this dataset directly to the `cases_dataset`.

!!!tips
    Apart from contrastive explanations, in the case of classification, the built-in [Projections](#projections) compute `targets` online and the `targets_dataset` is not necessary.

### Properties ###

- **search_method_class** (`Type[BaseSearchMethod]`): Abstract property to define the search method class to use. Must be implemented in subclasses. (see [Search Methods](#search-methods))
- **k** (`int`): Getter and setter for the `k` parameter.
- **returns** (`Union[List[str], str]`): Getter and setter for the `returns` parameter. Defines the elements to return in `self.explain()`.

### `explain(self, inputs, targets)` ###

Returns the relevant examples to explain the (inputs, targets). Projects inputs using `self.projection` and finds examples using the `self.search_method`.

- **inputs** (`Union[tf.Tensor, np.ndarray]`): Input samples to be explained. Shape: (n, ...) where n is the number of samples.
- **targets** (`Optional[Union[tf.Tensor, np.ndarray]]`): Targets associated with the `inputs` for projection. Shape: (n, nb_classes) where n is the number of samples and nb_classes is the number of classes. Not used in all projection. Used in contrastive methods to know the predicted classes of the provided samples.

**Returns:** Dictionary with elements listed in `self.returns`.

!!!info
    The `__call__` method is an alias for the `explain` method.

## Projections ##
Projections are functions that map input samples to a search space where examples are retrieved with a `search_method`. The search space should be relevant for the model (e.g. projecting the inputs into the latent space of the model).

!!!info
    If one decides to use the identity function as a projection, the search space will be the input space, thus rather explaining the dataset than the model.

The `Projection` class is a base class for projections. It involves two parts: `space_projection` and `weights`. The samples are first projected to a new space and then weighted. 

!!!warning
    If both parts are `None`, the projection acts as an identity function. In general, we advise that one part should involve the model to ensure meaningful distance calculations with respect to the model.

To know more about projections and their importance, you can refer to the [Projections](../../projections/) section.

## Search Methods ##

!!!info
    The search methods are hidden to the user and only used internally. However, they help to understand how the API works.

Search methods are used to retrieve examples from the `cases_dataset` that are relevant to the input samples.

!!!warning
    In an search method, the `cases_dataset` is the dataset that has been projected with a `Projection` object (see the previous section). The search methods are used to find examples in this projected space.

Each example-based method has its own search method. The search method is defined in the `search_method_class` property of the `ExampleMethod` class.

[^1]: [Natural Example-Based Explainability: a Survey (2023)](https://arxiv.org/abs/2309.03234)