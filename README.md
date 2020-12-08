<div align="center">
    <img src="./docs/assets/typo.png" width="25%" alt="Xplique" align="center" />
</div>
<br>

<div align="center">
    <a href="https://travis-ci.com/fel-thomas/xplique">
        <img alt="Build Status" src="https://travis-ci.com/fel-thomas/xplique.svg?token=R9xr216LTFpJW3LYYCaM&branch=master">
    </a>
</div>
<br>

**Xplique** is a Python module dedicated to explainability. It provides several submodules to learn
more about your tensorflow models (≥2.1). The three main submodules are _Attributions Methods_,
_Explainability Metrics_ and _Feature Visualization_ tools.

The _Attributions Method_ submodule implements various methods, with explanations, examples and 
links to official papers.

Soon, the _Explainability Metrics_ submodule will implement the current metrics related to 
explainability. These evaluations used in conjunction with the attribution methods allow to measure
the quality of the explanations.

Soon, the _Feature Visualization_ submodule will allow to represent neurons, channels or layers
by maximizing an input. 

The package is released under [MIT license](https://choosealicense.com/licenses/mit).

![Example of Attributions Methods results](./docs/assets/samples.png)

## Contents

- [Install](#installing) <br>
- [Get started](#get-started) <br>
- [Core features](#core-features) <br>
    - [Attributions Methods](#methods) <br>
    - [Concept based Methods](#concept-based-methods) <br>
    - [Metrics](#metrics) <br>
    - [Feature Visualization](#feature-visualization) <br>
- [Notebooks](#notebooks) <br>

## Installing

The library has been tested on Linux, MacOSX and Windows and relies on the following Python modules:

* Tensorflow (>=2.1)
* Numpy (>=1.18)

You can install Xplique using pip with:

```bash
pip install xplique
```

## Getting Started

let's start with a simple example, by computing Grad-CAM for several images (or a complete dataset)
on a trained model.

```python
from xplique.attributions import GradCAM

# load images, labels and model
# ...

method = GradCAM(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [Using the attributions methods](https://gist.github.com/napolar/c02cef48ae7fc20e76d633f3f1588c63)
<sub> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/napolar/c02cef48ae7fc20e76d633f3f1588c63/sample-generation.ipynb) </sub>

## Core features

### Attributions Methods

* [x] Deconvolution          [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/deconvnet.md)               [📄<sup>arxiv</sup>](https://arxiv.org/abs/1311.2901)
* [x] Grad-CAM               [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/grad_cam.md)                [📄<sup>arxiv</sup>](https://arxiv.org/abs/1610.02391)
* [x] Grad-CAM++             [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/grad_cam_pp.md)             [📄<sup>arxiv</sup>](https://arxiv.org/abs/1710.11063)
* [x] Gradient Input         [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/gradient_input.md)          [📄<sup>arxiv</sup>](https://arxiv.org/abs/1711.06104)
* [x] Guided Backprop        [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/guided_backpropagation.md)  [📄<sup>arxiv</sup>](https://arxiv.org/abs/1412.6806)
* [x] Integrated Gradients   [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/integrated_gradients.md)    [📄<sup>arxiv</sup>](https://arxiv.org/abs/1703.01365)
* [x] Occlusion              [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/occlusion.md)               [📄<sup>arxiv</sup>](https://arxiv.org/abs/1311.2901)
* [x] Rise                   [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/rise.md)                    [📄<sup>arxiv</sup>](https://arxiv.org/abs/1806.07421)   
* [x] Saliency               [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/saliency.md)                [📄<sup>arxiv</sup>](https://arxiv.org/abs/1312.6034)
* [x] SmoothGrad             [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/smoothgrad.md)              [📄<sup>arxiv</sup>](https://arxiv.org/abs/1706.03825)
* [x] SquareGrad             [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/square_grad.md)             [📄<sup>arxiv</sup>](https://arxiv.org/abs/1806.10758)
* [x] VarGrad                [ 📚<sup>Api</sup> ](https://napolar.github.io/xplique/api/vargrad.md)                 [📄<sup>arxiv</sup>](https://arxiv.org/abs/1810.03292)
* [ ] Ablation-CAM  
* [ ] Xray

### Concept-based Methods

* [x] [ Concept Activation Vector ](./api/cav.md)[^12]
* [x] [ Testing with Concept Activation Vector ](./api/tcav.md)[^12]
* [ ] Robust TCAV 
* [ ] Automatic Concept Extraction  

### Metrics

* [ ] Aocp  
* [ ] Fidelity correlation
* [ ] Irof     
* [ ] Pixel Flipping
* [ ] Stability

### Feature Visualization

* [ ] Vanilla


