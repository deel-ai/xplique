"""
NLP Util Classes
"""

from abc import ABC, abstractmethod
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from flair.models import SequenceTagger
from flair.data import Sentence

from ..types import Union, List, Tuple

class TokenExtractor(ABC):
    """Base class for the tokenizers.
    """
    def __init__(self):
        self.separator = " "

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
    """
    def __init__(self):
        super().__init__()
        self.separator = ". "

    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Extract tokens from a sentence, using the NLTK sentence tokenizer.

        Parameters
        ----------
        sentence : Union[List[str], str]
            The input sentence, or a list of input sentences.

        Returns
        -------
        tokens
            A list of extracted sentences.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        words = sent_tokenize(sentence)
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

class ClauseExtractor(TokenExtractor):
    """Uses Flair sentence tokenizer.
    """

    tagger = SequenceTagger.load("flair/chunk-english")

    def __init__(self, clause_type : List[str] = None):
        super().__init__()
        self.clause_type = clause_type

    def extract_tokens(self, sentence: Union[List[str], str]) -> List[str]:
        """
        Separates the input texts into clauses, and only keeps the ones belonging to
        the specified types.
        If clause_type is None, the texts are split but all the clauses are kept.

        Parameters
        ----------
        sentence
            A list of strings that we wish to separate into clauses.
        clause_type
            A list with the types of clauses to keep. Each clause shall be a string
            corresponding to one of Flair SequenceTagger dictionnary.
            Example: clause_type = ['NP', 'ADJP']
            If None, all clauses are kept.

        Returns
        -------
        clause_list
            A list with input texts split into clauses.
        """
        if not isinstance(sentence, str):
            sentence = '.'.join(sentence)
        sentence = Sentence(sentence)
        self.tagger.predict(sentences = sentence)
        clause_list = []
        for segment in sentence.get_labels():
            if self.clause_type is None:
                clause_list.append(segment.data_point.text)
            elif segment.value in self.clause_type:
                clause_list.append(segment.data_point.text)

        return clause_list, self.separator

class ExtractorFactory():
    """
    Factory for extractor classes.
    """
    @staticmethod
    def get_extractor(extract_fct = "clause", clause_type = None):
        """
        Get an instance of an extractor based on the specified extraction function.

        Parameters
        ----------
        extract_fct
            The type of extraction function to use, either 'word', 'sentence', 'excerpt', 'clause'.
            Default is "clause".
        clause_type
            Additional parameter specifying the type of clause (default is None).

        Returns
        -------
        Extractor
            An instance of the selected extractor based on the specified function.
        """
        if extract_fct == "clause":
            return ClauseExtractor(clause_type)
        if extract_fct == "sentence":
            return SentenceExtractor()
        if extract_fct == "word":
            return WordExtractor()
        if extract_fct == "excerpt":
            return ExcerptExtractor()
        raise ValueError("Extraction function can be only 'clause', \
                         'sentence', 'word' or 'excerpt'")
