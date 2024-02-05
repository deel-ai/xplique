"""
Custom pytorch operations
"""

from abc import ABC, abstractmethod
from math import ceil
import numpy as np
import torch

from ..types import Union, Tuple, Callable, List, Dict


class NlpPreprocessor(ABC):
    """
    Abstract base class for NLP preprocessing.

    Parameters
    ----------
    tokenizer
        The tokenizer function to be used for tokenization. It must be
        a callable that returns outputs matching the model inputs expectations.
    device
        The device on which to perform tokenization (default is 'cuda').
    kwargs
        A list of key value arguments that will be passed to the tokenizer when
        tokenizing inputs.
    """

    def __init__(self, tokenizer: Callable, device='cuda', **kwargs):
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_kwargs = kwargs

    def tokenize(self,
                 samples: Union[List[str], np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Transforms a list of strings into tokens for consumption by the transformer model.

        Parameters
        ----------
        samples
            A list of input strings to be tokenized.

        Returns
        -------
        tokenized
            Result of the tokenizer operation over the input sequences. Usually a dictionary
            with keys 'input_ids' and 'attention_mask'.
        """
        tokenized = self.tokenizer(
            list(samples),
            **self.tokenizer_kwargs
        ).to(self.device)

        return tokenized

    @abstractmethod
    def preprocess(self,
                   inputs: List[str],
                   labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-process the dataset and adapt it according to a specific task.

        Parameters
        ----------
        inputs
            The input text data.
        labels
            The corresponding labels.

        Returns:
        ----------
        preprocessed_inputs
            The pre-processed inputs.
        preprocessed_labels
            The pre-processed outputs.
        """
        raise NotImplementedError


def batcher(elements, batch_size: int):
    """
    An function to create batches from a list of elements.

    Parameters
    ----------
    elements
        The list of elements to be batched.
    batch_size
        The size of each batch.

    Returns
    ------
    batch
        A batch of elements (yielded).
    """
    nb_batchs = ceil(len(elements) / batch_size)

    for batch_i in range(nb_batchs):
        batch_start = batch_i * batch_size
        batch_end = batch_start + batch_size

        batch = elements[batch_start:batch_end]
        yield batch


def nlp_batch_predict(model: Union[torch.nn.Module, Callable],
                      preprocessor: NlpPreprocessor,
                      inputs: List[str],
                      labels: List[str],
                      batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-processes and predicts using the transformer model in batches.

    Parameters
    ----------
    model
        The transformer model for prediction.
    preprocessor
        An instance of NlpPreprocessor used for preprocessing the input texts.
    inputs
        A list of n_samples input texts to be predicted.
    labels
        A list of labels corresponding to the input texts.
    batch_size
        The batch size (default is 64).

    Returns
    -------
    predictions
        The predictions output of the model for the pre-processed input texts.
    processed_labels
        The pre-processed labels.
    """
    predictions = None
    processed_labels = None

    with torch.no_grad():
        dataset = list(zip(inputs, labels))
        for batch in batcher(dataset, batch_size):
            batch_inputs, batch_labels = zip(*batch)
            x_preprocessed, y_preprocessed = preprocessor.preprocess(
                np.array(batch_inputs), np.array(batch_labels))
            out_batch = model(**x_preprocessed)
            predictions = out_batch if predictions is None else torch.cat(
                [predictions, out_batch])
            processed_labels = y_preprocessed if processed_labels is None else torch.cat([
                processed_labels, y_preprocessed])

    return predictions, processed_labels
