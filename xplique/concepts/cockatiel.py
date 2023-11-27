"""
COCKATIEL Module for Pytorch
"""

from typing import Callable, List, Tuple, Optional, Union, Type
from math import ceil
import re

import numpy as np
import torch
from sklearn.decomposition import NMF
from IPython.display import display, HTML

from ..commons import TokenExtractor, WordExtractor, ExcerptExtractor, ExtractorFactory
from .craft import BaseCraft, Factorization, Sensitivity, MaskSampler
from .craft_torch import BaseCraftTorch

from xplique.attributions.base import BlackBoxExplainer

class CockatielTorch(BaseCraftTorch):
    """
    A class implementing COCKATIEL, the concept based explainability method for NLP introduced
    in https://arxiv.org/abs/2305.06754

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Pytorch model accepting data of shape (n_samples, channels, height, width).
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Pytorch model.
    tokenizer
        A callable object to transform strings into inputs for the model.
    number_of_concepts
        The number of concepts to extract. Default is 25.
    batch_size
        The batch size for all the operations that use the model. Default is 256.
    device
        The type of device on which to place the torch tensors
    """

    def __init__(
            self,
            input_to_latent_model: Callable,
            latent_to_logit_model: Callable,
            tokenizer: Callable,
            number_of_concepts: int = 25,
            batch_size: int = 256,
            device: str = 'cuda'
    ):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts, batch_size, device=device)
        self.tokenizer = tokenizer
        self.patch_extractor = ExcerptExtractor()

    def tokenize(self, samples: List[str]) -> torch.Tensor:
        """
        A function to transform a list of strings into tokens to be consumed
        by the transformer model.

        Parameters
        ----------
        samples : List[str]
            The list of strings to be tokenized.

        Returns
        -------
        tokens : torch.Tensor
            The tokenized representation of the input strings.
        """
        tokens = self.tokenizer(
            [sample for sample in samples],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        return tokens

    def _latent_predict(self, inputs: torch.Tensor, resize=None) -> torch.Tensor:
        """
        Compute the embedding space using the 1st model `input_to_latent_model`.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, channels, height, width).

        Returns
        -------
        activations
            The latent activations of shape (n_samples, height, width, channels)
        """
        def gen_activations():
            with torch.no_grad():
                nb_batchs = ceil(len(inputs) / self.batch_size)
                for batch_id in range(nb_batchs):
                    batch_start = batch_id * self.batch_size
                    batch_end = batch_start + self.batch_size
                    batch_tokenized = self.tokenize(
                        inputs[batch_start:batch_end])

                    batch_activations = self.input_to_latent_model(
                        **batch_tokenized)
                    yield batch_activations

        activations = torch.cat(list(gen_activations()), 0)
        return activations

    def _preprocess(self, activations: torch.Tensor) -> np.ndarray:
        """
        Preprocesses the activations to make sure that they're the right shape
        for being input to the NMF algorithm later.

        Parameters
        ----------
        activations
            The (non-negative) activations from the model under study.

        Returns
        -------
        activations
            The preprocessed activations, ready for COCKATIEL.
        """
        assert torch.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            # activations: (N, H, W, R)
            activations = torch.mean(activations, dim=(1, 2))

        return self._to_np_array(activations)

    def _extract_patches(self, inputs: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Extract patches (excerpts) from the input sentences, and compute their embeddings.

        Parameters
        ----------
        inputs
            Input sentences.

        Returns
        -------
        patches
            A list of excerpts (n_patches).
        activations
            The excerpts activations (n_patches, channels).
        """
        crops, _ = self.patch_extractor.extract_tokens(inputs)

        activations = self._latent_predict(crops)

        # applying GAP(.) on the activation and ensure positivity if needed
        activations = self._preprocess(activations)

        return crops, activations

    def estimate_importance(self,
                            inputs: np.ndarray = None,
                            sampler: MaskSampler = MaskSampler.SOBOL,
                            nb_design: int = 32) -> np.ndarray:
        """
        Estimates the importance of each concept for a given class, either globally
        on the whole dataset provided in the fit() method (in this case, inputs shall
        be set to None), or locally on a specific input sentence.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data on which to compute the importances.
            If None, then the inputs provided in the fit() method
            will be used (global importance of the whole dataset).
            Default is None.
        sampler
            The sampling method to use for masking. Default to MaskSampler.SOBOL.
        nb_design
            The number of design to use for the importance estimation. Default is 32.

        Returns
        -------
        importances
            The Sobol total index (importance score) for each concept.

        """
        return super().estimate_importance(inputs=inputs, sampler=sampler, nb_design=nb_design)

    def get_best_excerpts_per_concept(self,
                                      nb_excerpts: int = 10,
                                      nb_most_important_concepts: int = None) -> List[str]:
        """
        Return the best excerpts for each concept.

        Parameters
        ----------
        nb_excerpts
            The number of excerpts (patches) to fetch per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.

        Returns
        -------
        best_excerpt_per_concept
            The list of the best excerpts per concept
        """
        most_important_concepts = self.sensitivity.most_important_concepts
        if nb_most_important_concepts is not None:
            most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        best_crops_per_concept = []
        for c_id in most_important_concepts:
            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_excerpts]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]
            best_crops_per_concept.append(best_crops)
        return best_crops_per_concept

    def display_concepts_excerpts(self,
                                  nb_crops: int = 10,
                                  nb_most_important_concepts: int = None) -> None:
        """
        Display the best excerpts for each concept.

        Parameters
        ----------
        nb_crops
            The number of crops (patches) to display per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.
        """
        for c_id, c_id_importance, best_crops in \
            self._gen_best_concepts_crops(nb_crops, nb_most_important_concepts):

            print(f"Concept {c_id} has an importance value of "
                  f"{c_id_importance:.2f}")
            for crop in best_crops:
                print(f"\t{crop}")

    def plot_concept_attribution_maps(self,
                                      sentences: List[str],
                                      token_extractor: TokenExtractor,
                                      explainer_class: Type[BlackBoxExplainer],
                                      ignore_words: List[str] = None,
                                      importances: np.ndarray = None,
                                      nb_most_important_concepts: int = 5,
                                      title: str = "",
                                      filter_percentile: int = 80,
                                      display_width: int = 400) -> List[str]:
        """
        Display the concepts attribution maps for the sentences given in argument.

        Parameters
        ----------
        sentences
            The list of sentences for which attribution maps will be displayed.
        token_extractor
            The token extractor used to extract tokens from the sentences.
        explainer_class
            Explainer class used during the masking process. Typically a NlpOcclusion class.
        ignore_words
            Words to ignore during the occlusion process. These will not be part of the
            extracted concepts.
        importances
            The importances computed by the estimate_importance() method.
            If None is provided, then the global importances will be used, otherwise
            the local importances set in this parameter will be used.
        nb_most_important_concepts
            Number of most important concepts to consider. Default is 5.
        title
            The title to use when displaying the results.
        filter_percentile
            Percentile used to filter the concept heatmap when displaying the HTML
            sentences (only show concept if excess N-th percentile). Default is 80.
        display_width
            Width of the displayed HTML text. Default is 400.

        Returns
        -------
        html_lines
            List of HTML text lines.
        """
        if importances is None:
            # global
            most_important_concepts = self.sensitivity.most_important_concepts
        else:
            # local
            most_important_concepts = np.argsort(importances)[::-1]

        most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        # Build the CSS
        css = build_concepts_css(concepts_ids=most_important_concepts,
                                 cmaps=self.sensitivity.cmaps[:nb_most_important_concepts])
        html_output = [css]

        # Build the title
        html_output.append(f"<h3>{title}</h3>")

        # Build the legend
        html_output.append(
            f"<div class='legend_container'><div><em>Legend : </em></div><div>" \
            f"{get_legend(concepts_ids=most_important_concepts)}</div></div>")

        # Generate the HTML for each sentence
        html_output.append('<ul>') # list start
        for sentence in sentences:
            html_output.append('<li>')
            html_output.append(
                self.sentence_attribution_map(sentence=str(sentence),
                                              token_extractor = token_extractor,
                                              ignore_words = ignore_words,
                                              explainer_class = explainer_class,
                                              most_important_concepts = most_important_concepts,
                                              filter_percentile = filter_percentile,
                                              display_width = display_width))
            html_output.append('</li>')

        html_output.append('</ul>') # list end

        # Display the HTML text & return it as well
        display(HTML("".join(html_output)))
        return html_output

    def sentence_attribution_map(self,
                                 sentence: str,
                                 token_extractor: WordExtractor,
                                 explainer_class: type[BlackBoxExplainer],
                                 most_important_concepts: np.ndarray,
                                 ignore_words: List[str] = None,
                                 filter_percentile: int = 80,
                                 display_width: int = 400) -> str:
        """
        Compute the concepts attribution maps for the sentence given in argument,
        and return the corresponding sentence as an HTML formatted sentence where
        the words belonging to each concept are highlighted with colors.

        Parameters
        ----------
        sentences
            The sentence to process.
        token_extractor
            The token extractor used to extract tokens from the sentences.
        explainer_class
            Explainer class used during the masking process. Typically a NlpOcclusion class.
        most_important_concepts
            The concepts ids to display.
        ignore_words
            Words to ignore during the occlusion process. These will not be part of the
            extracted concepts.
        filter_percentile
            Percentile used to filter the concept heatmap when displaying the HTML
            sentences (only show concept if excess N-th percentile). Default is 80.
        display_width
            Width of the displayed HTML text. Default is 400.

        Returns
        -------
        html_lines
            The sentence formatted as HTML text.
        """
        extracted_words, separator = token_extractor.extract_tokens(sentence)

        # Filter words
        if ignore_words is not None:
            words = [word for word in extracted_words if word not in ignore_words]
        else:
            words = extracted_words

        explainer = explainer_class(model = self.transform)
        l_importances = explainer.explain(sentence = sentence, words = words, separator = separator)

        # Display only the most important concepts
        l_importances = l_importances[most_important_concepts]

        return convert_sentence_to_html(extracted_words, separator, words, l_importances,
                                        concepts_ids = most_important_concepts,
                                        filter_percentile = filter_percentile,
                                        display_width = display_width)

def get_legend(concepts_ids: List[int]):
    """
    Generates an HTML legend for the concepts attribution maps.

    Parameters
    ----------
    concepts_ids
        The list of concepts ids that are handled by the current Cockatiel instance.

    Returns
    -------
    html_legend
        The legend formatted as HTML text.
    """
    labels = [f"Concept {c_id}" for c_id in concepts_ids]
    legend_importances = np.eye(len(concepts_ids)) / 2.0
    return convert_sentence_to_html(extracted_words=labels,
                                    words=labels,
                                    separator=" ",
                                    concepts_ids=concepts_ids,
                                    explanation=legend_importances)


def convert_sentence_to_html(
        extracted_words: List[str],
        separator: str,
        words: List[str],
        explanation: np.ndarray,  # shape: (2 x 7) : nb_concepts x nb_words
        concepts_ids: List[int],  # id of each concept
        filter_percentile: int = 80,
        display_width:int = 400  # px
) -> str:
    """
    Generates the visualization for COCKATIEL's explanations.

    Parameters
    ----------
    extracted_words
        List of extracted words from the input sentence: it shall
        be possible to reconstruct the whole input sentence using `extracted_words`
        and `separator`.
    separator
        Separator used to join extracted_words to build the whole sentence.
    words
        List of words used to generate the explanation, these words should be part
        of the sentence and are attached to a concept of the `explanation` parameter below.
    explanation
        An array that corresponds to the output of the occlusion function.
        Has a shape of (nb_concepts x nb_words).
    concepts_ids
        The list of concepts ids that are handled by the current Cockatiel instance.
    filter_percentile
        Percentile used to filter the concept heatmap when displaying the HTML
        sentences (only show concept if excess N-th percentile). Default is 80.
    display_width
        Width of the displayed HTML text. Default is 400.

    Returns
    -------
    html_lines
        The sentence formatted as HTML text.
    """
    l_phi = np.array(explanation)

    # Filter the values that are below the percentile for each concept importance array
    sigmas = [[np.percentile(phi, filter_percentile)] for phi in l_phi]
    l_phi = l_phi * np.array(l_phi > sigmas, np.float32)

    phi_html = []

    # Build a dictionary of structure 'word' => importance_value, concept_id
    words_importances = dict(
        zip(words, zip(l_phi.max(axis=0), l_phi.argmax(axis=0))))

    for word in extracted_words:
        if word in words_importances:
            value, value_max_id = words_importances[word]
            concept_id = concepts_ids[value_max_id]
            title = f"[{word}] concept:{concept_id} importance:{value:.2f}"
            if value > 0:
                phi_html.append(f"<span class='concept{concept_id}' "
                                f"style='--opacity:{value:.2f}' "
                                f"title='{title}' "
                                f">{word}{separator}</span>")
            else:
                phi_html.append(
                    f"<span title='{title}'>{word}{separator}</span>")
        else:
            # ignored words
            phi_html.append(f"<span> {word}{separator}</span>")

    html_result = f"<span style='display: flex; width: {display_width}px; flex-wrap: wrap'>" \
        + " ".join(phi_html) + "</span>"
    return html_result


def build_concepts_css(concepts_ids: List[str], cmaps: List[str]):
    """
    Generates the CSS for HTML COCKATIEL's explanations.

    Parameters
    ----------
    concepts_ids
        The list of concepts ids.
    cmaps
        The color associated with each concept.
        Can be either:
            - A tuple (r, g, b) of color to use as a base for the colormap.
            - A colormap name compatible with `plt.get_cmap(cmap)`.
    """
    legend_css = """
        <style>
            .legend_container {
                display: flex;
                align-items: center;
            }
        </style>
    """

    concepts_css = []
    for c_id, cmap in zip(concepts_ids, cmaps):
        red, green, blue, alpha = [int(c*255) for c in cmap(1)]
        txt = f"\n.concept{c_id} {{" \
            f"  padding: 1px 5px;" \
            f"  border: solid 3px;" \
            f"  border-color: rgba({red}, {green}, {blue});" \
            f"  border-radius: 10px;" \
            f"  background-color: rgba({red}, {green}, {blue}, var(--opacity));" \
            f"  --opacity: {alpha};" \
            f"}}"

        concepts_css.append(txt)
    css_text = "<style>span {padding: 4px 1px;}" + \
        " ".join(concepts_css) + "</style>"

    return legend_css + css_text
