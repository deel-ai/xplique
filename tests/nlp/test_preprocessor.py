import numpy as np

import torch
import pytest
from transformers import RobertaTokenizerFast
from xplique.commons.torch_operations import NlpPreprocessor, batcher, nlp_batch_predict


class ImdbPreprocessor(NlpPreprocessor):
    def preprocess(self, inputs: np.ndarray, labels: np.ndarray):
        preprocessed_inputs = self.tokenize(samples=inputs.tolist())
        preprocessed_labels = torch.Tensor(np.array(
            labels.tolist()) == 'positive').int().to(self.device)
        return preprocessed_inputs, preprocessed_labels


@pytest.fixture
def imdb_preprocessor():
    pretrained_model_path = "wrmurray/roberta-base-finetuned-imdb"
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_path)
    return ImdbPreprocessor(tokenizer,
                            device='cpu',
                            padding="max_length",
                            max_length=512,
                            truncation=True,
                            return_tensors='pt')


def test_imdb_preprocessor(imdb_preprocessor):
    inputs = np.array(['This is an example sentence.',
                      'Another sentence for testing.'])
    labels = np.array(['positive', 'negative'])
    x_preprocessed, y_preprocessed = imdb_preprocessor.preprocess(
        inputs, labels)
    assert 'input_ids' in x_preprocessed
    assert 'attention_mask' in x_preprocessed
    assert x_preprocessed['input_ids'].shape == x_preprocessed['attention_mask'].shape == (
        2, 512)
    assert y_preprocessed.shape == (2,)
    assert y_preprocessed[0] == 1
    assert y_preprocessed[1] == 0


def test_batcher():
    # Test case 1: elements perfectly divisible by batch_size
    elements1 = [1, 2, 3, 4, 5, 6]
    batch_size1 = 2
    batches1 = list(batcher(elements1, batch_size1))
    assert batches1 == [[1, 2], [3, 4], [5, 6]]

    # Test case 2: elements not perfectly divisible by batch_size
    elements2 = [1, 2, 3, 4, 5]
    batch_size2 = 2
    batches2 = list(batcher(elements2, batch_size2))
    assert batches2 == [[1, 2], [3, 4], [5]]

    # Test case 3: batch_size greater than the length of elements
    elements3 = [1, 2, 3, 4, 5]
    batch_size3 = 10
    batches3 = list(batcher(elements3, batch_size3))
    assert batches3 == [[1, 2, 3, 4, 5]]

    # Test case 4: batch_size equal to the length of elements
    elements4 = [1, 2, 3, 4, 5]
    batch_size4 = 5
    batches4 = list(batcher(elements4, batch_size4))
    assert batches4 == [[1, 2, 3, 4, 5]]

    # Test case 5: with a complex list of elements
    elements5 = list(zip([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e']))
    batch_size5 = 2
    batches5 = list(batcher(elements5, batch_size5))
    assert batches5 == [[(1, 'a'), (2, 'b')], [(3, 'c'), (4, 'd')], [(5, 'e')]]


class ImdbClassifier(torch.nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes

    def forward(self, **kwargs):
        return torch.randn(kwargs['input_ids'].shape[0], self.nb_classes)


def test_batch_predict(imdb_preprocessor):
    inputs = ['text1', 'text2', 'text3']
    labels = ['positive', 'negative', 'positive']

    batch_size = 2
    nb_classes = 2
    mock_model = ImdbClassifier(nb_classes)

    predictions, processed_labels = nlp_batch_predict(
        mock_model, imdb_preprocessor, inputs, labels, batch_size)

    assert len(predictions) == len(processed_labels) == len(inputs)
