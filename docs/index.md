<h1>
    <p align="center">
      <img alt="Xplique logo" src="./assets/typo.png" width="200px">
    </p>
</h1>

<p align="center">
    <a href="https://travis-ci.com/napolar/xplique">
        <img alt="Build Status" src="https://travis-ci.com/napolar/xplique.svg?token=R9xr216LTFpJW3LYYCaM&branch=master">
    </a>
</p>

Xplique is a Python module that provide fast implementations of the latest methods for 
interpretability of neural networks. This package focuses on performance in order to generate 
explanations on large datasets. These methods are currently developped using Tensorflow AutoGraph.
The package is released under [MIT license](https://choosealicense.com/licenses/mit).

![Sample results](./assets/samples.png)

## Contents

[Install](#installing) <br>
[Get started](#get-started) <br>
[Examples](#examples) <br>
[API Reference](#api) <br>

---

## Implemented methods

* [Saliency](./api/saliency.md) <br>
* [DeconvNet](./api/deconvnet.md) <br>
* [Guided Backpropagation](./api/guided_backprop.md) <br>
* [Gradient Input](./api/gradient_input.md) <br>
* [Occlusion Sensitivity](./api/occlusion_sensitivity.md) <br>
* [Integrated Gradient](./api/integrated_gradients.md) <br>
* [SmoothGrad](./api/smoothgrad.md) <br>
* [Grad-CAM](./api/grad_cam.md) <br>
* [Grad-CAM++](./api/grac_cam_pp.md) <br>



## Installing

The library has been tested on Linux, MacOSX and Windows and relies on the following Python modules:

* Tensorflow (>=2.1)
* Numpy (>=1.18)
* opencv-python (>=4.1.0)

You can install Xplique using pip with:

```bash
pip install xplique
```

## Get started

let's start with a simple example, by computing Grad-CAM and Saliency maps for several images
(or a complete dataset) on a trained model.

```python
from xplique.methods import GradCAM, Saliency

# load images, labels and model
# ...

saliency = Saliency(model)
saliency_maps = saliency(images, labels)

gradcam = GradCAM(model)
gradcam_maps = gradcam(images, labels) 

print(gradcam_maps.shape) # ndarray with the same size as images 
```

## Examples

[Generating samples using the different methods available](https://gist.github.com/napolar/c02cef48ae7fc20e76d633f3f1588c63)
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/napolar/c02cef48ae7fc20e76d633f3f1588c63/sample-generation.ipynb)

