"""
NLP Util Classes
"""

from abc import ABC, abstractmethod
import re
from typing import Type

from nltk.tokenize import word_tokenize, sent_tokenize

from flair.models import SequenceTagger
from flair.data import Sentence

from spacy import Language

from ..types import Union, List, Tuple


class TokenExtractor(ABC):
    """
    Base class for the token extractors.

    Parameters
    ----------
    language
        The language of the sentence.
    """

    def __init__(self, language: str = "english"):
        self.separator = " "
        self.language = language

    @abstractmethod
    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Extract tokens from a sentence.

        Parameters
        ----------
        sentence
            The input sentence, or a list of input sentences.

        Returns
        -------
        tokens
            A list of extracted tokens.
        """


class WordExtractor(TokenExtractor):
    """
    Uses NLTK word tokenizer.
    """

    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Extract tokens from a sentence, using the NLTK word tokenizer

        Parameters
        ----------
        sentence
            The input sentence, or a list of input sentences.

        Returns
        -------
        tokens
            A list of extracted words.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        words = word_tokenize(sentence)

        return words, self.separator


class SentenceExtractor(TokenExtractor):
    """
    Uses NLTK sentence tokenizer.

    Parameters
    ----------
        language
        The language of the sentence.
    """

    def __init__(self, language: str = "english"):
        super().__init__(language=language)
        self.separator = ". "

    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Extract tokens from a sentence, using the NLTK sentence tokenizer.

        Parameters
        ----------
        sentence
            The input sentence, or a list of input sentences.

        Returns
        -------
        tokens
            A list of extracted sentences.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        words = sent_tokenize(sentence, self.language)
        return words, self.separator


class ExcerptExtractor(TokenExtractor):
    """
    Uses a custom excerpt tokenizer.
    """

    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Extract excerpts from a sentence:
            - split the input string with '.', '?', and '!' separators
            - ignore excerpts not starting with a capital letter

        Parameters
        ----------
        sentence
            The input sentence, or a list of input sentences.

        Returns
        -------
        tokens
            A list of extracted excerpts.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        # Split the input text with '.', '?' and '!' separators.
        regexp = r"[A-Z][^.?!]*[.?!]"
        res = re.findall(regexp, sentence)
        excerpt_dataset = [st.strip() for st in res]
        return excerpt_dataset, self.separator


class FlairClauseExtractor(TokenExtractor):
    """
    Uses Flair sentence tokenizer.

    Parameters
    ----------
    tagger
        A trained Flair SequenceTagger.
    clause_type
        A list with the types of clauses to keep. Each clause shall be a string
        corresponding to one of Flair SequenceTagger dictionnary.
        Example: clause_type = ['NP', 'ADJP']
        If None, all clauses are kept.
    language
        The language of the sentence.
    """

    def __init__(self,
                 tagger: Type[SequenceTagger],
                 clause_type: List[str] = None,
                 language: str = "english"):
        super().__init__(language=language)
        self.clause_type = clause_type
        self.tagger = tagger

    def extract_tokens(self, sentence: Union[List[str], str]) -> Tuple[List[str], str]:
        """
        Separates the input texts into clauses, and only keeps the ones belonging to
        the specified types.
        If clause_type is None, the texts are split but all the clauses are kept.

        Parameters
        ----------
        sentence
            A list of strings that we wish to separate into clauses.

        Returns
        -------
        clause_list
            A list with input texts split into clauses.
        separator
            The separator used for spliting the sentence.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        sentence = Sentence(sentence)
        self.tagger.predict(sentences=sentence)
        clause_list = []
        for segment in sentence.get_labels():
            if self.clause_type is None:
                clause_list.append(segment.data_point.text)
            elif segment.value in self.clause_type:
                clause_list.append(segment.data_point.text)

        return clause_list, self.separator


class SpacyClauseExtractor(TokenExtractor):
    """
    Uses Spacy sentence tokenizer.

    Parameters
    ----------
    pipeline
        A trained Spacy pipeline.
    clause_type
        A list with the types of clauses to keep. Each clause shall be a string
        corresponding to one of Spacy Tag dictionnary.
        Example: clause_type = ['NNP', 'VBZ']
        If None, all clauses are kept.
    """

    def __init__(self,
                 pipeline: Type[Language],
                 clause_type: List[str] = None):
        super().__init__(language=None)
        self.clause_type = clause_type
        self.pipeline = pipeline

    def extract_tokens(self, sentence: Union[List[str], str]) -> Tuple[List[str], str]:
        """
        Separates the input texts into clauses, and only keeps the ones belonging to
        the specified types.
        If clause_type is None, the texts are split but all the clauses are kept.

        Parameters
        ----------
        sentence
            A list of strings that we wish to separate into clauses.

        Returns
        -------
        clause_list
            A list with input texts split into clauses.
        separator
            The separator used for spliting the sentence.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)

        clause_list = []
        for token in self.pipeline(sentence):
            if self.clause_type is None:
                clause_list.append(token.text)
            elif token.tag_ in self.clause_type:
                clause_list.append(token.text)

        return clause_list, self.separator


class ExtractorFactory():
    """
    Factory for extractor classes.
    """
    @staticmethod
    def get_extractor(extract_fct="sentence", tagger=None, pipeline=None, clause_type=None):
        """
        Get an instance of an extractor based on the specified extraction function.

        Parameters
        ----------
        extract_fct
            The type of extraction function to use, either 'word', 'sentence', 'excerpt',
            'flair_clause', 'spacy_clause'.
            Default is "sentence".
        tagger
            A Flair SequenceTagger to use for Flair clause extractor.
        pipeline
            A Spacy Pipeline to use for Spacy clause extractor.
        clause_type
            Additional parameter specifying the type of clause (default is None).

        Returns
        -------
        Extractor
            An instance of the selected extractor based on the specified function.
        """
        if extract_fct == "flair_clause":
            return FlairClauseExtractor(tagger, clause_type)
        if extract_fct == "spacy_clause":
            return SpacyClauseExtractor(pipeline, clause_type)
        if extract_fct == "sentence":
            return SentenceExtractor()
        if extract_fct == "word":
            return WordExtractor()
        if extract_fct == "excerpt":
            return ExcerptExtractor()
        raise ValueError("Extraction function can be only 'clause', \
                         'sentence', 'word' or 'excerpt'")
