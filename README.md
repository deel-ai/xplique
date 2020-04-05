<div align="center">
    <img src="./docs/assets/typo.png" width="25%" alt="Xplique" align="center" />
</div>
<br>
<br>

[![Build Status](https://travis-ci.com/napolar/xplique.svg?token=R9xr216LTFpJW3LYYCaM&branch=master)](https://travis-ci.com/napolar/xplique)

<br>

**Xplique is a Python module that provide fast implementations of the latest methods for 
interpretability of neural networks. This package focuses on performance in order to generate 
explanations on large datasets. These methods are currently developped using Tensorflow AutoGraph.
The package is released under [MIT license](https://choosealicense.com/licenses/mit).**



## Implemented methods

Saliency <br>
Gradient Input <br>
Integrated Gradient <br>
SmoothGrad <br>
Grad-CAM <br>


## Installing

The library has been tested on Linux, MacOSX and Windows and relies on the following Python modules:

* Tensorflow (>=2.1)
* Numpy (>=1.18)
* opencv-python (>=4.1.0)

You can install Xplique using pip with:

```bash
pip install xplique
```

## Examples

let's start with a simple example, by computing Grad-CAM for several images (or a complete dataset)
on a trained model.

```python
from xplique.methods import GradCAM

# load images, labels and model
# ...

method = GradCAM(model)
explanations = method.explain(images, labels)
```