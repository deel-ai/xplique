# Grad-CAM++

Grad-CAM++ is a technique for producing visual explanations that can be used on Convolutional Neural
Netowrk (CNN) which uses both gradients and the feature maps of the last convolutional layer.

## Example

```python
from xplique.attributions import GradCAMPP

# load images, labels and model
# ...

method = GradCAMPP(model, conv_layer=-3)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**GradCam ++**: Going Further](https://colab.research.google.com/drive/1NRzdZdwxEYhC3_0gf8VpC_bg4YQcVsnO) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NRzdZdwxEYhC3_0gf8VpC_bg4YQcVsnO) </sub>

{{xplique.attributions.grad_cam_pp.GradCAMPP}}

[^1]: [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks (2017).](https://arxiv.org/abs/1710.11063)
