<div align="center">
    <img src="./assets/banner_white.png#only-dark" width="75%" alt="Xplique banner" align="center" />
    <img src="./assets/banner.png#only-light" width="75%" alt="Xplique banner" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9, 3.10-efefef">
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
    <a href="https://pepy.tech/project/xplique">
        <img alt="Pepy" src="https://static.pepy.tech/badge/xplique">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<div class="central">
    🦊 <b>Xplique</b> (pronounced <i>\ɛks.plik\</i>) is a Python toolkit dedicated to explainability. The goal of this library is to gather the state of the art of Explainable AI to help you understand your complex neural network models. Originally built for Tensorflow's model it also works for Pytorch's model partially.

  <br>
  <a href="https://deel-ai.github.io/xplique/"><strong>Explore Xplique docs »</strong></a>
  <br>
  <br>
  <a href="api/attributions/api_attributions/">Attributions</a>
  ·
  <a href="api/concepts/cav/">Concept</a>
  ·
  <a href="api/feature_viz/feature_viz/">Feature Visualization</a>
  ·
  <a href="api/metrics/api_metrics/">Metrics</a>
</div>

The library is composed of several modules, the _Attributions Methods_ module implements various methods (e.g Saliency, Grad-CAM, Integrated-Gradients...), with explanations, examples and links to official papers.
The _Feature Visualization_ module allows to see how neural networks build their understanding of images by finding inputs that maximize neurons, channels, layers or compositions of these elements.
The _Concepts_ module allows you to extract human concepts from a model and to test their usefulness with respect to a class.
Finally, the _Metrics_ module covers the current metrics used in explainability. Used in conjunction with the _Attribution Methods_ module, it allows you to test the different methods or evaluate the explanations of a model.

<p align="center" width="100%">
    <img width="95%" src="./assets/modules.png">
</p>

<br>

## 🔥 Tutorials

??? example "We propose some Hands-on tutorials to get familiar with the library and its api"

    - [**Attribution Methods**: Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>

    <p align="center" width="100%">
        <a href="https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2">
            <img width="95%" src="assets/attributions.jpeg">
        </a>
    </p>

    - [**Attribution Methods**: Sanity checks paper](https://colab.research.google.com/drive/1uJOmAg6RjlOIJj6SWN9sYRamBdHAuyaS) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uJOmAg6RjlOIJj6SWN9sYRamBdHAuyaS) </sub>
    - [**Attribution Methods**: Tabular data and Regression](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub>
    - [**FORGRad**: Gradient strikes back with FORGrad](https://colab.research.google.com/drive/1ibLzn7r9QQIEmZxApObowzx8n9ukinYB) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ibLzn7r9QQIEmZxApObowzx8n9ukinYB) </sub>
    - [**Attribution Methods**: Metrics](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg) </sub>

    <p align="center" width="100%">
        <a href="https://colab.research.google.com/drive/1WEpVpFSq-oL1Ejugr8Ojb3tcbqXIOPBg"> 
            <img width="95%" src="assets/metrics.jpeg">
        </a>
    </p>

    - [**Concepts Methods**: Testing with Concept Activation Vectors](https://colab.research.google.com/drive/1iuEz46ZjgG97vTBH8p-vod3y14UETvVE) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iuEz46ZjgG97vTBH8p-vod3y14UETvVE) </sub>

    <p align="center" width="100%">
        <a href="https://colab.research.google.com/drive/1iuEz46ZjgG97vTBH8p-vod3y14UETvVE">
            <img width="95%" src="assets/concepts.jpeg">
        </a>
    </p>

    - [**Feature Visualization**: Getting started](https://colab.research.google.com/drive/1st43K9AH-UL4eZM1S4QdyrOi7Epa5K8v) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1st43K9AH-UL4eZM1S4QdyrOi7Epa5K8v) </sub>

    - [**Feature Visualization**: Getting started](https://colab.research.google.com/drive/1st43K9AH-UL4eZM1S4QdyrOi7Epa5K8v) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1st43K9AH-UL4eZM1S4QdyrOi7Epa5K8v) </sub>
    <p align="center" width="100%">
        <a href="https://colab.research.google.com/drive/1st43K9AH-UL4eZM1S4QdyrOi7Epa5K8v"> 
            <img width="95%" src="assets/feature_viz.jpeg">
        </a>
    </p>
    - [**Modern Feature Visualization with MaCo**: Getting started](https://colab.research.google.com/drive/1l0kag1o-qMY4NCbWuAwnuzkzd9sf92ic) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l0kag1o-qMY4NCbWuAwnuzkzd9sf92ic) </sub>

    You can find a certain number of [other practical tutorials just here](tutorials/). This section is actively developed and more contents will be
    included. We will try to cover all the possible usage of the library, feel free to contact us if you have any suggestions or recommandations towards tutorials you would like to see.


## 🚀 Quick Start

Xplique requires a version of python higher than 3.6 and several libraries including Tensorflow and Numpy. Installation can be done using Pypi:

```python
pip install xplique
```

Now that Xplique is installed, here are 4 basic examples of what you can do with the available modules.

??? example "Attributions Methods"
    Let's start with a simple example, by computing Grad-CAM for several images (or a complete dataset) on a trained model.

    ```python
    from xplique.attributions import GradCAM

    # load images, labels and model
    # ...

    explainer = GradCAM(model)
    explanations = explainer.explain(images, labels)
    # or just `explainer(images, labels)`
    ```

    All attributions methods share a common API. You can find out more about it [here](api/attributions/api_attributions/).

    In addition, you should also look at the [model's specificities](api/attributions/model/) and the [operator parameter documentation](api/attributions/operator/)

??? example "Attributions Metrics"

    In order to measure if the explanations provided by our method are faithful (it reflects well the functioning of the model) we can use a fidelity metric such as Deletion

    ```python
    from xplique.attributions import GradCAM
    from xplique.metrics import Deletion

    # load images, labels and model
    # ...

    explainer = GradCAM(model)
    explanations = explainer(inputs, labels)
    metric = Deletion(model, inputs, labels)

    score_grad_cam = metric(explanations)
    ```

    All attributions metrics share a common API. You can find out more about it [here](api/metrics/api_metrics/).

??? example "Concepts Extraction"

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

    More information on CAV [here](api/concepts/cav/) and on TCAV [here](api/concepts/tcav/).

??? example "Feature Visualization"

    Finally, in order to find an image that maximizes a neuron and at the same time a layer, we build two objectives that we combine together. We then call the optimizer which returns our images

    ```python
    from xplique.features_visualizations import Objective
    from xplique.features_visualizations import optimize

    # load a model...

    neuron_obj = Objective.neuron(model, "logits", 200)
    channel_obj = Objective.layer(model, "mixed3", 10)

    obj = neuron_obj + 2.0 * channel_obj
    images, obj_names = optimize(obj)
    ```

    Want to know more ? Check the Feature Viz [documentation](api/feature_viz/feature_viz/)


## 📦 What's Included

??? abstract "Table of attributions available"

    All the attributions method presented below handle both **Classification** and **Regression** tasks.

    | **Attribution Method** | Type of Model | Source                                    | Tabular Data       | Images             | Time-Series        | Tutorial           |
    | :--------------------- | :------------ | :---------------------------------------- | :----------------: | :----------------: | :----------------: | :----------------: |
    | Deconvolution          | TF            | [Paper](https://arxiv.org/abs/1311.2901)  | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
    | Grad-CAM               | TF            | [Paper](https://arxiv.org/abs/1610.02391) |                    | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
    | Grad-CAM++             | TF            | [Paper](https://arxiv.org/abs/1710.11063) |                    | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
    | Gradient Input         | TF, Pytorch**            | [Paper](https://arxiv.org/abs/1704.02685) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
    | Guided Backprop        | TF            | [Paper](https://arxiv.org/abs/1412.6806)  | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
    | Integrated Gradients   | TF, Pytorch**       | [Paper](https://arxiv.org/abs/1703.01365) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo) |
    | Kernel SHAP            | TF, Pytorch** , Callable*     | [Paper](https://arxiv.org/abs/1705.07874) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) |
    | Lime                   | TF, Pytorch** , Callable*     | [Paper](https://arxiv.org/abs/1602.04938) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) |
    | Occlusion              | TF, Pytorch** , Callable*     | [Paper](https://arxiv.org/abs/1311.2901)  | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q) |
    | Rise                   | TF, Pytorch** , Callable*     | [Paper](https://arxiv.org/abs/1806.07421) | WIP                | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO) |
    | Saliency               | TF, Pytorch**            | [Paper](https://arxiv.org/abs/1312.6034)  | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
    | SmoothGrad             | TF, Pytorch**            | [Paper](https://arxiv.org/abs/1706.03825) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) |
    | SquareGrad             | TF, Pytorch**            | [Paper](https://arxiv.org/abs/1806.10758) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) |
    | VarGrad                | TF, Pytorch**            | [Paper](https://arxiv.org/abs/1810.03292) | ✔                  | ✔                 | WIP                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) |
    | Sobol Attribution      | TF, Pytorch**            | [Paper](https://arxiv.org/abs/2111.04138) |                    | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) |
    | Hsic Attribution      | TF, Pytorch**            | [Paper](https://arxiv.org/abs/2206.06219) |                    | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) |
    | FORGrad enhancement      | TF, Pytorch**            | [Paper](https://arxiv.org/abs/2307.09591) |                    | ✔                 |                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ibLzn7r9QQIEmZxApObowzx8n9ukinYB) |

    TF : Tensorflow compatible

    \* : See the [Callable documentation](callable/)

    ** : See the [Xplique for Pytorch documentation](pytorch/), and the [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe)<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) </sub> notebook


??? abstract "Table of attribution's metric available"

    | **Attribution Metrics** | Type of Model | Property         | Source                                    |
    | :---------------------- | :------------ | :--------------- | :---------------------------------------- |
    | MuFidelity              | TF, Pytorch** | Fidelity         | [Paper](https://arxiv.org/abs/2005.00631) |
    | Deletion                | TF, Pytorch** | Fidelity         | [Paper](https://arxiv.org/abs/1806.07421) |
    | Insertion               | TF, Pytorch** | Fidelity         | [Paper](https://arxiv.org/abs/1806.07421) |
    | Average Stability       | TF, Pytorch** | Stability        | [Paper](https://arxiv.org/abs/2005.00631) |
    | MeGe                    | TF, Pytorch** | Representativity | [Paper](https://arxiv.org/abs/2009.04521) |
    | ReCo                    | TF, Pytorch** | Consistency      | [Paper](https://arxiv.org/abs/2009.04521) |
    | (WIP) e-robustness      |

    TF : Tensorflow compatible

    ** : See the [Xplique for Pytorch documentation](pytorch/), and the [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe)<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) </sub> notebook

??? abstract "Table of concept methods available"

    | **Concepts method**             | Type of Model | Source                                        |
    | :------------------------------ | :------------ | :-------------------------------------------- |
    | Concept Activation Vector (CAV) | TF            | [Paper](https://arxiv.org/pdf/1711.11279.pdf) |
    | Testing CAV (TCAV)              | TF            | [Paper](https://arxiv.org/pdf/1711.11279.pdf) |
    | (WIP) Robust TCAV               |               |
    | (WIP) Automatic Concept Extraction (ACE)        |

    TF : Tensorflow compatible

??? abstract "Table of Feature Visualization methods available"

    | **Feature Visualization** [(Paper)](https://distill.pub/2017/feature-visualization/) | Type of Model | Details                                                                                                            |
    | :----------------------------------------------------------------------------------- | :------------ | :----------------------------------------------------------------------------------------------------------------- |
    | Neurons                                                                              | TF            | Optimizes for specific neurons                                                                              |
    | Layer                                                                                | TF            | Optimizes for specific layers                                                                               |
    | Channel                                                                              | TF            | Optimizes for specific channels                                                                             |
    | Direction                                                                            | TF            | Optimizes for specific vector                                                                               |
    | Fourrier Preconditioning                                                             | TF            | Optimize in Fourier basis (see [preconditioning](https://distill.pub/2017/feature-visualization/#preconditioning)) |
    | Objective combination                                                                | TF            | Allows to combine objectives                                                                                       |
    | MaCo                                                                                 | TF            | Fixed Magnitude optimisation, see [Paper](https://arxiv.org/pdf/2306.06805.pdf)                                                                                       |

    TF : Tensorflow compatible


## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Xplique toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](contributing/).

## 👀 See Also

This library is one approach of many to explain your model. We don't expect it to be the perfect
 solution; we create it to explore one point in the space of possibilities.

??? info "Other interesting tools to explain your model:"

    - [Lucid](https://github.com/tensorflow/lucid) the wonderful library specialized in feature visualization from OpenAI.
    - [Captum](https://captum.ai/) the Pytorch library for Interpretability research
    - [Tf-explain](https://github.com/sicara/tf-explain) that implement multiples attribution methods and propose callbacks API for tensorflow.
    - [Alibi Explain](https://github.com/SeldonIO/alibi) for model inspection and interpretation
    - [SHAP](https://github.com/slundberg/shap) a very popular library to compute local explanations using the classic Shapley values from game theory and their related extensions

??? info "To learn more about Explainable AI in general:"

    - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) by Christophe Molnar.
    - [Interpretability Beyond Feature Attribution](https://www.youtube.com/watch?v=Ff-Dx79QEEY) by Been Kim.
    - [Explaining ML Predictions: State-of-the-art, Challenges, and Opportunities](https://www.youtube.com/watch?v=7dpOSmQ89L8) by Himabindu Lakkaraju, Julius Adebayo and Sameer Singh.
    - [A Roadmap for the Rigorous Science of Interpretability](https://www.youtube.com/watch?v=MMxZlr_L6YE) by Finale Doshi-Velez.
    - [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of explainability for this purpose

??? info "More from the DEEL project:"

    - [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
    - [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
    - [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
    - [LARD](https://github.com/deel-ai/LARD) Landing Approach Runway Detection (LARD) is a dataset of aerial front view images of runways designed for aircraft landing phase
    - [PUNCC](https://github.com/deel-ai/puncc) Puncc (Predictive uncertainty calibration and conformalization) is an open-source Python library that integrates a collection of state-of-the-art conformal prediction algorithms and related techniques for regression and classification problems
    - [OODEEL](https://github.com/deel-ai/oodeel) OODeel is a library that performs post-hoc deep OOD detection on already trained neural network image classifiers. The philosophy of the library is to favor quality over quantity and to foster easy adoption
    - [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.


## 🙏 Acknowledgments

<img align="right" src="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10#only-dark" width="25%" alt="DEEL Logo" />
<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png#only-light" width="25%" alt="DEEL Logo" />
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creators

This library was started as a side-project by [Thomas FEL](http://thomasfel.fr) who is
currently a graduate student at the Artificial and Natural Intelligence Toulouse Institute under
the direction of [Thomas SERRE](https://serre-lab.clps.brown.edu). His thesis work focuses on
explainability for deep neural networks.

He then received help from some members of the <a href="https://www.deel.ai/"> DEEL </a> team
to enhance the library namely from [Lucas Hervier](https://github.com/lucashervier) and [Antonin Poché](https://github.com/AntoninPoche).


## 🗞️ Citation

If you use Xplique as part of your workflow in a scientific publication, please consider citing the 🗞️ [Xplique official paper](https://arxiv.org/abs/2206.04394):

```
@article{fel2022xplique,
  title={Xplique: A Deep Learning Explainability Toolbox},
  author={Fel, Thomas and Hervier, Lucas and Vigouroux, David and Poche, Antonin and Plakoo, Justin and Cadene, Remi and Chalvidal, Mathieu and Colin, Julien and Boissin, Thibaut and Bethune, Louis and Picard, Agustin and Nicodeme, Claire 
          and Gardes, Laurent and Flandin, Gregory and Serre, Thomas},
  journal={Workshop on Explainable Artificial Intelligence for Computer Vision (CVPR)},
  year={2022}
}
```

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
