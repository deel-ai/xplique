"""
Module related to Occlusion sensitivity method for NLP.
"""

import numpy as np

from .base import BlackBoxExplainer
from ..commons import Tasks
from ..types import Callable, Union, Optional, OperatorSignature, List

class NlpOcclusion(BlackBoxExplainer):
    """
    Occlusion class for NLP.
    """
    def __init__(self,
                 model: Callable,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None):
        super().__init__(model, batch_size, operator)

    @staticmethod
    def _get_masks(input_len: int) -> np.ndarray:
        """
        Generate occlusion masks for a given input length.

        Parameters
        ----------
        input_len : int
            The length of the input for which occlusion masks are generated.
            Typically it will be the number of words of a sentence.

        Returns
        -------
        occlusion_masks : np.ndarray
            The boolean occlusion masks, an identity matrix with False for the main diagonal.
            This kind of mask can be used to generate n sentences,
            each with a single distinct word removed.
        """
        return np.eye(input_len) == 0

    @staticmethod
    def _apply_masks(words: List[str], masks: np.ndarray) -> np.ndarray:
        """
        Apply occlusion masks to a list of words.

        Parameters
        ----------
        words : List[str]
            The list of words to which occlusion masks are applied.
        masks : np.ndarray
            The boolean occlusion masks to be applied.

        Returns
        -------
        occluded_words : np.ndarray
            The list of words with occlusion masks applied.
        """
        perturbated_words = [np.array(words)[mask].tolist() for mask in masks]
        return perturbated_words

    def explain(self,
                sentence: str,
                words: List[str],
                separator: str) -> np.ndarray:
        """
        Generate an explanation for the input sentence, by providing the importance of each word.
        The importance will be computed by successively occluding each word of the sentence and
        studying the impact of this occlusion on the model results.

        Parameters
        ----------
        sentence : str
            The input sentence for which an explanation is generated.
        words : List[str]
            List of words used to generate the explanation. These words must be part of
            the input sentence, the importance will be computed on this list of words
            (i.e some words of the original sentence can be omited this way).
        separator : str
            The separator used to join the words after the occlusion step, so a full
            sentence can be fed to the model.

        Returns
        -------
        explanation : np.ndarray
            The generated explanation of format (nb_concepts, nb_words).
        """

        # generate n sentences with a different word masked (removed) each time
        masks = NlpOcclusion._get_masks(len(words))
        perturbated_words = NlpOcclusion._apply_masks(words, masks)

        perturbated_sentences = [sentence]
        perturbated_sentences.extend(
            [separator.join(perturbated_word) for perturbated_word in perturbated_words])

        # transform the perturbated reviews into their concept representation
        # u_values has shape: ((W+1) x C)
        u_values = self.model(perturbated_sentences)

        # Compute sensitivities: importances = u_value of the whole sentence - u_value of each word
        whole_sentence_uvalues = u_values[0,:]
        words_uvalues = u_values[1:,:]
        l_importances = (whole_sentence_uvalues - words_uvalues).transpose()
        l_importances /= (np.max(np.abs(l_importances)) + 1e-5)

        return l_importances
