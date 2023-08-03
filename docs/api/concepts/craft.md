# CRAFT

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>
[View colab Tensorflow tutorial](https://colab.research.google.com/drive/1jmyhb89Bdz7H4G2KfK8uEVbSC-C_aht_) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>
[View colab Pytorch tutorial](https://colab.research.google.com/drive/16Jn2pQy4gi2qQYZFnuW6ZNtVAYiNyJHO) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/concepts/craft.py) |
📰 [Paper](https://arxiv.org/pdf/2211.10154)

CRAFT or Concept Recursive Activation FacTorization for Explainability is a method for automatically extracting human-interpretable concepts from deep networks.

This concept activations factorization method aims to explain a trained model's decisions on a per-class and per-image basis by highlighting both "what" the model saw and “where” it saw it. Thus CRAFT generates post-hoc local and global explanations.
 
It is made up from 3 ingredients:

1. a method to recursively decompose concepts into sub-concepts
2. a method to better estimate the importance of extracted concepts
3. a method to use any attribution method to create concept attribution maps, using implicit differentiation

CRAFT requires splitting the model in two parts: $(g, h)$ such that $f(x) = (g \cdot h)(x)$. To put it simply, $g$ is the function that maps our input to the latent space (an inner layer of our model), and $h$ is the function that maps the latent space to the output.
The concepts will be extracted from this latent space.

!!!info
    It is important to note that if the model contains a global average pooling layer, it is strongly recommended to provide CRAFT with the layer before the global average pooling.

!!!warning
    Please keep in mind that the activations must be positives (after relu or any positive activation function)


## Example

Use Craft to investigate a single class.

```python
from xplique.concepts import CraftTf as Craft

# Cut the model in two parts (as explained in the paper)
# first part is g(.) our 'input_to_latent' model returning positive activations,
# second part is h(.) our 'latent_to_logit' model

g = tf.keras.Model(model.input, model.layers[-3].output)
h = tf.keras.Model(model.layers[-2].input, model.layers[-1].output)

# Create a Craft concept extractor from these 2 models
craft = Craft(input_to_latent_model = g,
              latent_to_logit_model = h,
              number_of_concepts = 10,
              patch_size = 80,
              batch_size = 64)

# Use Craft to get the crops (crops), the embedding of the crops (crops_u),
# and the concept bank (w)
crops, crops_u, w = craft.fit(images_preprocessed, class_id=rabbit_class_id)

# Compute Sobol indices to understand which concept matters
importances = craft.estimate_importance()

# Display those concepts by showing the 10 best crops for each concept
craft.plot_concepts_crops(nb_crops=10)

```

Use CraftManager to investigate multiple classes.

```python
from xplique.concepts import CraftManagerTf as CraftManager


# Cut the model in two parts (as explained in the paper)
# first part is g(.) our 'input_to_latent' model returning positive activations,
# second part is h(.) our 'latent_to_logit' model

g = tf.keras.Model(model.input, model.layers[-3].output)
h = tf.keras.Model(model.layers[-2].input, model.layers[-1].output)

# CraftManager will create one instance of Craft per class of interest
# to investigate
list_of_class_of_interest = [0, 491, 497, 569, 574] # list of class_ids
cm = CraftManager(input_to_latent_model = g,
                 latent_to_logit_model = h,
                 inputs = inputs_preprocessed,
                 labels = y,
                 list_of_class_of_interest = list_of_class_of_interest,
                 number_of_concepts = 10,
                 patch_size = 80,
                 batch_size = 64)

cm.fit(nb_samples_per_class=50)

# Compute Sobol indices to understand which concept matters
cm.estimate_importance()

# Display those concepts by showing the 10 best crops for each concept,
# for the 1st class
cm.plot_concepts_crops(class_id=0, nb_crops=10)

```


{{xplique.concepts.craft_tf.CraftTf}}

{{xplique.concepts.craft_tf.CraftManagerTf}}


[^1]: [CRAFT: Concept Recursive Activation FacTorization for Explainability (2023).](https://arxiv.org/pdf/2211.10154.pdf)
