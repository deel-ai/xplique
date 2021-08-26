<div align="center">
    <img src="docs/assets/banner.png" width="75%" alt="Xplique" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="https://github.com/deel-ai/xplique/actions/workflows/python-lints.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/xplique/actions/workflows/python-lints.yml/badge.svg">
    </a>
    <a href="https://github.com/deel-ai/xplique/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/deel-ai/xplique/actions/workflows/python-tests.yml/badge.svg">
    </a>
     <a href="https://github.com/deel-ai/xplique/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/deel-ai/xplique/actions/workflows/python-publish.yml/badge.svg">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<p align="center">
    🦊 <b>Xplique</b> (pronounced <i>\ɛks.plik\</i>) is a Python toolkit dedicated to explainability, currently based on Tensorflow.
    The goal of this library is to gather the state of the art of Explainable AI to help you understand your complex neural network models.

  <br>
  <a href="https://deel-ai.github.io/xplique/"><strong>Explore Xplique docs »</strong></a>
  <br>
  <br>
  <a href="https://deel-ai.github.io/xplique/attributions">Attributions</a>
  ·
  <a href="https://deel-ai.github.io/xplique/concepts">Concept</a>
  ·
  <a href="https://deel-ai.github.io/xplique/feature_visualization">Feature Visualization</a>
  ·
  <a href="https://deel-ai.github.io/xplique/metrics">Metrics</a>
</p>

The library is composed of several modules, the _Attributions Methods_ module implements various methods (e.g Saliency, Grad-CAM, Integrated-Gradients...), with explanations, examples and links to official papers.
The _Feature Visualization_ module allows to see how neural networks build their understanding of images by finding inputs that maximize neurons, channels, layers or compositions of these elements.
The _Concepts_ module allows you to extract human concepts from a model and to test their usefulness with respect to a class.
Finally, the _Metrics_ module covers the current metrics used in explainability. Used in conjunction with the _Attribution Methods_ module, it allows you to test the different methods or evaluate the explanations of a model.

<p align="center" width="100%">
    <img width="33%" src="./docs/assets/attributions.png"> 
    <img width="39%" src="./docs/assets/feature_viz.png"> 
    <img width="23.4%" src="./docs/assets/concepts.png"> 
</p>

<br>

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🚀 Quick Start](#-quick-start)
- [🔥 Tutorials](#-tutorials)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [📝 License](#-license)

## 🚀 Quick Start

Xplique requires a version of python higher than 3.6 and several libraries including Tensorflow and Numpy. Installation can be done using Pypi:

```python
pip install xplique
```

Now that Xplique is installed, here are 4 basic examples of what you can do with the available modules.

### Attributions Methods

let's start with a simple example, by computing Grad-CAM for several images (or a complete dataset) on a trained model.

```python
from xplique.attributions import GradCAM

# load images, labels and model
# ...

explainer = GradCAM(model)
explanations = explainer.explain(images, labels)
# or just `explainer(images, labels)`
```

### Attributions Metrics

In order to measure if the explanations provided by our method are faithful (it reflects well the functioning of the model) we can use a fidelity metric such as Deletion

```python
from xplique.attributions import GradCAM
from xplique.metrics import Deletion

# load images, labels and model
# ...

explainer = GradCAM(model)
metric = Deletion(model, inputs, labels)

score_grad_cam = metric(explainer)
```

### Concepts Extraction

Concerning the concept-based methods, we can for example extract a concept vector from a layer of a model. In order to do this, we use two datasets, one containing inputs containing the concept: `positive_samples`, the other containing other entries which do not contain the concept: `negative_samples`.

```python
from xplique.concepts import Cav

# load a model, samples that contain a concept
# (positive) and samples who don't (negative)
# ...

extractor = Cav(model, 'mixed3')
concept_vector = extractor(positive_samples,
                           negative_samples)
```

### Feature Visualization

Finally, in order to find an image that maximizes a neuron and at the same time a layer, we build two objectives that we combine together. We then call the optimizer which returns our images

```python
from xplique.features_visualizations import Objective
from xplique.features_visualizations import optimize

# load a model...

neuron_obj = Objective.neuron(model, "logits", 200)
channel_obj = Objective.layer(model, "mixed3", 10)

obj = neuron_obj + 2.0 * channel_obj
images = optimize(obj)
```

## 🔥 Tutorials

We propose some Hands-on tutorials to get familiar with the library and its api:

- **Getting Started**    
  <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11Hiu5SMER7dfug3th2EBNRiKHTVGfJxB?authuser=1) </sub>
- **Tabular Data and Regression**    
  <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq?authuser=1) </sub>
- **Introduction to Metrics**    
  <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12OtJKdUuuaiRVpe-luhScv368gLeuT58?authuser=1) </sub>

You can find a certain number of other practical tutorials [just here](https://github.com/deel-ai/xplique/blob/master/TUTORIALS.md). This section is actively developed and more contents will be
included. We will try to cover all the possible usage of the library, feel free to contact us if you have any suggestions or recommandations towards tutorials you would like to see.

## 📦 What's Included

All the attributions method presented below handle both **Classification** and **Regression** tasks.

| **Attribution Method** | Type of Model | Source                                    | Tabular Data       | Images             | Time-Series        |
| :--------------------- | :------------ | :---------------------------------------- | :----------------: | :----------------: | :----------------: |
| Deconvolution          | TF            | [Paper](https://arxiv.org/abs/1311.2901)  | ✔                  | ✔                 | TBD                |
| Grad-CAM               | TF            | [Paper](https://arxiv.org/abs/1610.02391) |                    | ✔                  | TBD                |
| Grad-CAM++             | TF            | [Paper](https://arxiv.org/abs/1710.11063) |                    | ✔                  | TBD                |
| Gradient Input         | TF            | [Paper](https://arxiv.org/abs/1711.06104) | ✔                  | ✔                 | TBD                |
| Guided Backprop        | TF            | [Paper](https://arxiv.org/abs/1412.6806)  | ✔                  | ✔                 | TBD                |
| Integrated Gradients   | TF            | [Paper](https://arxiv.org/abs/1703.01365) | ✔                  | ✔                 | TBD                |
| Kernel SHAP            | Callable      | [Paper](https://arxiv.org/abs/1705.07874) | ✔                  | ✔                 | TBD                |
| Lime                   | Callable      | [Paper](https://arxiv.org/abs/1602.04938) | ✔                  | ✔                 | TBD                |
| Occlusion              | Callable      | [Paper](https://arxiv.org/abs/1311.2901)  | ✔                  | ✔                 | TBD                |
| Rise                   | Callable      | [Paper](https://arxiv.org/abs/1806.07421) | TBD                | ✔                  | TBD                |
| Saliency               | TF            | [Paper](https://arxiv.org/abs/1312.6034)  | ✔                  | ✔                 | TBD                |
| SmoothGrad             | TF            | [Paper](https://arxiv.org/abs/1706.03825) | ✔                  | ✔                 | TBD                |
| SquareGrad             | TF            | [Paper](https://arxiv.org/abs/1806.10758) | ✔                  | ✔                 | TBD                |
| VarGrad                | TF            | [Paper](https://arxiv.org/abs/1810.03292) | ✔                  | ✔                 | TBD                |

| **Attribution Metrics** | Type of Model | Property  | Source                                    |
| :---------------------- | :------------ | :-------- | :---------------------------------------- |
| MuFidelity              | TF            | Fidelity  | [Paper](https://arxiv.org/abs/2005.00631) |
| Deletion                | TF            | Fidelity  | [Paper](https://arxiv.org/abs/1806.07421) |
| Insertion               | TF            | Fidelity  | [Paper](https://arxiv.org/abs/1806.07421) |
| Average Stability       | TF            | Stability | [Paper](https://arxiv.org/abs/2005.00631) |
| (WIP) MeGe              |               |           |
| (WIP) ReCo              |               |           |

| **Concepts method**             | Type of Model | Source                                        |
| :------------------------------ | :------------ | :-------------------------------------------- |
| Concept Activation Vector (CAV) | TF            | [Paper](https://arxiv.org/pdf/1711.11279.pdf) |
| Testing CAV                     | TF            | [Paper](https://arxiv.org/pdf/1711.11279.pdf) |
| (WIP) Robust TCAV               |               |

| **Feature Visualization** [(Paper)](https://distill.pub/2017/feature-visualization/) | Type of Model | Details                                                                                                            |
| :----------------------------------------------------------------------------------- | :------------ | :----------------------------------------------------------------------------------------------------------------- |
| Neurons                                                                              | TF            | Optimizes for specific neurons                                                                              |
| Layer                                                                                | TF            | Optimizes for specific layers                                                                               |
| Channel                                                                              | TF            | Optimizes for specific channels                                                                             |
| Direction                                                                            | TF            | Optimizes for specific vector                                                                               |
| Fourrier Preconditioning                                                             | TF            | Optimize in Fourier basis (see [preconditioning](https://distill.pub/2017/feature-visualization/#preconditioning)) |
| Objective combination                                                                | TF            | Allows to combine objectives                                                                                       |

<div align="right">
  <i>methods with TF need a Tensorflow model. </i>
</div>

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Xplique toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](https://github.com/deel-ai/xplique/blob/master/CONTRIBUTING.md).

## 👀 See Also

This library is one approach of many to explain your model. We don't expect it to be the final solution; we create it to explore one point in the space of possibilities.

Other tools to explain your model include:

- [Lucid](https://github.com/tensorflow/lucid) the wonderful library specialized in feature visualization from OpenAI.
- [Captum](https://captum.ai/) the Pytorch library for Interpretability research
- [Tf-explain](https://github.com/sicara/tf-explain) that implement multiples attribution methods and propose callbacks API for tensorflow.
- [Alibi Explain](https://github.com/SeldonIO/alibi) for model inspection and interpretation
- [SHAP](https://github.com/slundberg/shap) a very popular library to compute local explanations using the classic Shapley values from game theory and their related extensions

To learn more about Explainable AI in general, see:

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) an excellent book by Christophe Molnar.
- [Interpretability Beyond Feature Attribution](https://www.youtube.com/watch?v=Ff-Dx79QEEY) by Been Kim.
- [A Roadmap for the Rigorous Science of Interpretability](https://www.youtube.com/watch?v=MMxZlr_L6YE) by Finale Doshi-Velez.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of explainability for this purpose

## 🙏 Acknowledgments

<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creator

This library was started as a side-project by [Thomas FEL](https://twitter.com/Napoolar) who is doing a thesis on explainability.

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
