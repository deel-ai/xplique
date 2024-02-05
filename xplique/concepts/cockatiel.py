"""
COCKATIEL Module for Pytorch
"""

from typing import Callable, List, Tuple, Optional, Union, Type
from math import ceil

import html
from IPython.display import display, HTML

import numpy as np
import torch

from xplique.attributions.base import BlackBoxExplainer
from ..commons import TokenExtractor, WordExtractor, ExcerptExtractor, NlpPreprocessor
from .craft import MaskSampler
from .craft_torch import BaseCraftTorch


class BaseCockatiel(BaseCraftTorch):
    """
    A class implementing COCKATIEL, the concept based explainability method for NLP introduced
    in https://arxiv.org/abs/2305.06754

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Pytorch model accepting tokenized inputs.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Pytorch model.
    preprocessor
        A callable object to transform strings into inputs for the model.
        Embeds a tokenizer that will be used to feed the inputs to the model.
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
            preprocessor: NlpPreprocessor,
            number_of_concepts: int = 25,
            batch_size: int = 256,
            device: str = 'cuda'
    ):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts, batch_size, device=device)
        self.preprocessor = preprocessor
        self.patch_extractor = ExcerptExtractor()

    def _latent_predict(self, inputs: List[str], resize=None) -> torch.Tensor:
        """
        Compute the embedding space using the 1st model `input_to_latent_model`.

        Parameters
        ----------
        inputs
            A list of input string data.
        resize
            Unused parameter.

        Returns
        -------
        activations
            The latent activations of shape (n_samples, channels)
        """
        def gen_activations():
            with torch.no_grad():
                nb_batchs = ceil(len(inputs) / self.batch_size)
                for batch_id in range(nb_batchs):
                    batch_start = batch_id * self.batch_size
                    batch_end = batch_start + self.batch_size
                    batch_tokenized = self.preprocessor.tokenize(
                        samples=inputs[batch_start:batch_end])

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

        # pylint disable=no-member
        assert torch.min(activations) >= 0.0, "Activations must be positive."

        if len(activations.shape) == 4:
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
                            nb_design: int = 32,
                            cmaps: Optional[Union[Tuple, str]] = None) -> np.ndarray:
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
            The sampling method to use for masking. Defaults to MaskSampler.SOBOL.
        nb_design
            The number of design to use for the importance estimation. Default is 32.
        cmaps
            The list of colors associated with each concept.
            Can be either:
                - A list of (r, g, b) colors to use as a base for the colormap.
                - A colormap name compatible with `plt.get_cmap(cmap)`.

        Returns
        -------
        importances
            The Sobol total index (importance score) for each concept.

        """
        return super().estimate_importance(inputs=inputs, sampler=sampler,
                                           nb_design=nb_design, cmaps=cmaps)

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
        best_excerpts_per_concept
            The list of the best excerpts per concept
        """
        best_excerpts_per_concept = []
        for _, _, best_crops in \
                self._gen_best_concepts_crops(nb_excerpts, nb_most_important_concepts):
            best_excerpts_per_concept.append(best_crops)
        return best_excerpts_per_concept


class CockatielNlpVisualizationMixin():
    """
    Class containing text visualization methods for Cockatiel.
    """

    def display_concepts_excerpts(self,
                                  nb_patchs: int = 10,
                                  nb_most_important_concepts: int = None) -> None:
        """
        Display the best excerpts for each concept.

        Parameters
        ----------
        nb_patchs
            The number of patches to display per concept. Defaults to 10.
        nb_most_important_concepts
            The number of concepts to display. If provided, only display
            nb_most_important_concepts, otherwise display them all.
            Default is None.
        """
        for c_id, c_id_importance, best_crops in \
                self._gen_best_concepts_crops(nb_patchs, nb_most_important_concepts):

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
                                      css_namespace: str = "",
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
        css_namespace
            A namespace to be prefixed to the css class in order to avoid collisions.
            Usefull when running several instances of Cockatiel, to prevent having the
            same color for all the concepts of the same name.
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
        concepts_css_names = [
            f'{css_namespace}_{concept_id}' for concept_id in most_important_concepts]

        css = CockatielNlpVisualizationMixin.build_concepts_css(
            concepts_names=concepts_css_names,
            cmaps=self.sensitivity.cmaps[:nb_most_important_concepts])
        html_output = [css]

        # Build the title
        html_output.append(f"{title}")

        # Build the legend
        html_output.append(
            f"<div class='legend_container'><div><em>Legend : </em></div><div>"
            f"{CockatielNlpVisualizationMixin.get_legend(concepts_css_names)}</div></div>")

        # Generate the HTML for each sentence
        html_output.append('<ul>')  # list start
        for sentence in sentences:
            html_output.append('<li>')
            html_output.append(
                self.sentence_attribution_map(sentence=str(sentence),
                                              token_extractor=token_extractor,
                                              ignore_words=ignore_words,
                                              explainer_class=explainer_class,
                                              most_important_concepts=most_important_concepts,
                                              concepts_names=concepts_css_names,
                                              filter_percentile=filter_percentile,
                                              display_width=display_width))
            html_output.append('</li>')

        html_output.append('</ul>')  # list end

        # Display the HTML text & return it as well
        display(HTML("".join(html_output)))
        return html_output

    def sentence_attribution_map(self,
                                 sentence: str,
                                 token_extractor: WordExtractor,
                                 explainer_class: Type[BlackBoxExplainer],
                                 most_important_concepts: np.ndarray,
                                 concepts_names: List[str] = None,
                                 ignore_words: List[str] = None,
                                 filter_percentile: int = 80,
                                 display_width: int = 400) -> str:
        """
        Compute the concepts attribution maps for the sentence given in argument,
        and return the corresponding sentence as an HTML formatted sentence where
        the words belonging to each concept are highlighted with colors.

        Parameters
        ----------
        sentence
            The sentence to process.
        token_extractor
            The token extractor used to extract tokens from the sentences.
        explainer_class
            Explainer class used during the masking process. Typically a NlpOcclusion class.
        most_important_concepts
            The concepts ids to display.
        concepts_names
            The concepts names to use for the HTML display, corresponding to
            most_important_concepts.
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

        explainer = explainer_class(model=self.transform)
        l_importances = explainer.explain(
            sentence=sentence, words=words, separator=separator)

        # Display only the most important concepts
        l_importances = l_importances[most_important_concepts]

        return self.convert_sentence_to_html(extracted_words, separator, words, l_importances,
                                             concepts_names=concepts_names,
                                             filter_percentile=filter_percentile,
                                             display_width=display_width)

    @staticmethod
    def get_legend(concepts_names: List[str]) -> str:
        """
        Generates an HTML legend for the concepts attribution maps.

        Parameters
        ----------
        concepts_names
            The list of concepts names that are handled by the current Cockatiel instance.

        Returns
        -------
        html_legend
            The legend formatted as HTML text.
        """

        # Extract the label name from the concept name, which are composed of [css_namespace]_[cid]
        labels = [
            f"Concept {c_name.split('_')[-1]}" for c_name in concepts_names]
        legend_importances = np.eye(len(concepts_names)) / 2.0
        return CockatielNlpVisualizationMixin.\
            convert_sentence_to_html(extracted_words=labels,
                                     words=labels,
                                     separator=" ",
                                     concepts_names=concepts_names,
                                     explanation=legend_importances)

    @staticmethod
    def convert_sentence_to_html(
            extracted_words: List[str],
            separator: str,
            words: List[str],
            explanation: np.ndarray,
            concepts_names: List[str],
            filter_percentile: int = 80,
            display_width: int = 400
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
        concepts_names
            The list of concepts names that are handled by the current Cockatiel instance.
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

        i = 0
        for word in extracted_words:
            if word in words_importances:
                value, value_max_id = words_importances[word]

                # concept name is composed of css_namespace and c_id
                concept_name = concepts_names[value_max_id]
                c_id = concept_name.split('_')[-1]

                title = f"[{html.escape(word)}] concept:{c_id} importance:{value:.2f}"

                # add debug infos in the popup
                # indices = np.nonzero(l_phi[:, i])[0]
                # title += "\n DEBUG:"
                # for j in indices:
                #     concept_name = concepts_names[j]
                #     title += f"\n concept:{concept_name} importance {l_phi[j,i]:.2f}"

                if value > 0:
                    phi_html.append(f"<span class='concept{concept_name}' "
                                    f"style='--opacity:{value:.2f}' "
                                    f"title='{title}' "
                                    f">{word}{separator}</span>")
                else:
                    phi_html.append(
                        f"<span title='{title}'>{word}{separator}</span>")

                i += 1
            else:
                # ignored words
                phi_html.append(f"<span> {word}{separator}</span>")

        html_result = f"<span style='display: flex; width: {display_width}px; flex-wrap: wrap'>" \
            + " ".join(phi_html) + "</span>"
        return html_result

    @staticmethod
    def build_concepts_css(concepts_names: List[str], cmaps: List[str]):
        """
        Generates the CSS for HTML COCKATIEL's explanations.

        Parameters
        ----------
        concepts_names
            The list of concepts names.
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
        for c_id, cmap in zip(concepts_names, cmaps):
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


class CockatielTorch(BaseCockatiel, CockatielNlpVisualizationMixin):
    """
    Class implementing Cockatiel on Pytorch, adapted for text visualization.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Pytorch model accepting tokenized inputs.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Pytorch model.
    preprocessor
        A callable object to transform strings into inputs for the model.
        Embeds a tokenizer that will be used to feed the inputs to the model.
    number_of_concepts
        The number of concepts to extract. Default is 25.
    batch_size
        The batch size for all the operations that use the model. Default is 256.
    device
        The type of device on which to place the torch tensors
    """
