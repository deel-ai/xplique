# Guided Backpropagation

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/guided_backpropagation.py) | 📰 [ See paper](https://arxiv.org/abs/1412.6806)

## Example

```python
from xplique.attributions import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**Guided Backprop**: Going Further](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1)

{{xplique.attributions.guided_backpropagation.GuidedBackprop}}
