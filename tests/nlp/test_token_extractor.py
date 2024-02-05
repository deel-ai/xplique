from xplique.commons.nlp import WordExtractor, SentenceExtractor
from xplique.commons.nlp import FlairClauseExtractor, ExcerptExtractor, ExtractorFactory, SpacyClauseExtractor
from flair.models import SequenceTagger

import spacy

import pytest


@pytest.fixture
def example_sentence():
    return "One two three. Second sentence.Third Sentence, test1, test2; test3-test4 .GO!"\
           " Trust me,. sentence not starting with capital letter."\
           " Sentence with dots..Word, Word, word,...word ....so a sentence"


@pytest.fixture
def chunk_english_tagger():
    return SequenceTagger.load("flair/chunk-english")


@pytest.fixture
def web_sm_pipeline():
    spacy.cli.download("en_core_web_sm")
    return spacy.load("en_core_web_sm")


def test_word_extractor(example_sentence):
    extractor = WordExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '
    expected_tokens = ['One', 'two', 'three', '.',
                       'Second', 'sentence.Third', 'Sentence', ',', 'test1', ',', 'test2', ';',
                       'test3-test4', '.GO', '!', 'Trust', 'me', ',', '.', 'sentence', 'not',
                       'starting', 'with', 'capital', 'letter', '.', 'Sentence', 'with', 'dots',
                       '..', 'Word', ',', 'Word', ',', 'word', ',', '...', 'word', '....', 'so',
                       'a', 'sentence']
    assert tokens == expected_tokens, print('tokens:', tokens)


def test_word_extractor_ignore_words(example_sentence):
    extractor = WordExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '
    expected_tokens = ['One', 'two', 'three', '.',
                       'Second', 'sentence.Third', 'Sentence', ',', 'test1', ',', 'test2', ';',
                       'test3-test4', '.GO', '!', 'Trust', 'me', ',', '.', 'sentence', 'not',
                       'starting', 'with', 'capital', 'letter', '.', 'Sentence', 'with', 'dots',
                       '..', 'Word', ',', 'Word', ',', 'word', ',', '...', 'word', '....',
                       'so', 'a', 'sentence']
    assert tokens == expected_tokens, print('tokens:', tokens)


def test_word_extractor_from_list(example_sentence):
    extractor = WordExtractor()
    tokens, separator = extractor.extract_tokens(
        [example_sentence, example_sentence])
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '


def test_sentence_extractor(example_sentence):
    extractor = SentenceExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == '. '
    expected_tokens = ['One two three.',
                       'Second sentence.Third Sentence, test1, test2; test3-test4 .GO!',
                       'Trust me,.',
                       'sentence not starting with capital letter.',
                       'Sentence with dots..Word, Word, word,...word ....so a sentence']
    assert tokens == expected_tokens, print('tokens:', tokens)


def test_excerpt_extractor(example_sentence):
    extractor = ExcerptExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '
    expected_tokens = ['One two three.',
                       'Second sentence.',
                       'Third Sentence, test1, test2; test3-test4 .',
                       'GO!',
                       'Trust me,.',
                       'Sentence with dots.',
                       'Word, Word, word,.']
    assert tokens == expected_tokens, print('tokens:', tokens)


def test_flair_clause_extractor_close_type_none(example_sentence, chunk_english_tagger):
    clause_extractor = FlairClauseExtractor(
        tagger=chunk_english_tagger, clause_type=None)
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = ['One two three',
                       'Second sentence.Third Sentence',
                       'test1',
                       'test2',
                       'test3-test4',
                       'GO',
                       'Trust',
                       'me',
                       'sentence',
                       'not starting',
                       'with',
                       'capital letter',
                       'Sentence',
                       'with',
                       'dots.',
                       'Word',
                       'Word, word,...word',
                       'a sentence']
    assert tokens == expected_tokens


def test_flair_clause_extractor_close_type_NP(example_sentence, chunk_english_tagger):
    clause_extractor = FlairClauseExtractor(
        tagger=chunk_english_tagger, clause_type=['NP'])
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = ['One two three',
                       'Second sentence.Third Sentence',
                       'test1',
                       'test2',
                       'test3-test4',
                       'me',
                       'sentence',
                       'capital letter',
                       'Sentence',
                       'dots.',
                       'Word',
                       'Word, word,...word',
                       'a sentence']
    assert tokens == expected_tokens


def test_clause_extractor_close_type_ADJP(example_sentence, chunk_english_tagger):
    clause_extractor = FlairClauseExtractor(
        tagger=chunk_english_tagger, clause_type=['ADJP'])
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    print(tokens)
    expected_tokens = []
    assert tokens == expected_tokens


def test_spacy_clause_extractor_close_type_none(example_sentence, web_sm_pipeline):
    clause_extractor = SpacyClauseExtractor(
        pipeline=web_sm_pipeline, clause_type=None)
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = ['One', 'two', 'three', '.', 'Second', 'sentence', '.',
                       'Third', 'Sentence', ',', 'test1', ',', 'test2', ';',
                       'test3', '-', 'test4', '.GO', '!', 'Trust', 'me', ',',
                       '.', 'sentence', 'not', 'starting', 'with', 'capital',
                       'letter', '.', 'Sentence', 'with', 'dots', '..', 'Word',
                       ',', 'Word', ',', 'word,', '...', 'word', '....', 'so', 'a', 'sentence']
    assert tokens == expected_tokens


def test_spacy_clause_extractor_close_type_NN(example_sentence, web_sm_pipeline):
    clause_extractor = SpacyClauseExtractor(
        pipeline=web_sm_pipeline, clause_type=['NN'])
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = ['sentence', 'test1', 'test3', 'test4', 'sentence',
                       'capital', 'letter', 'Sentence', '...', 'word', 'sentence']

    assert tokens == expected_tokens


def test_extractor_factory():
    word_extractor = ExtractorFactory.get_extractor(extract_fct="word")
    assert isinstance(word_extractor, WordExtractor)

    sentence_extractor = ExtractorFactory.get_extractor(extract_fct="sentence")
    assert isinstance(sentence_extractor, SentenceExtractor)

    excerpt_extractor = ExtractorFactory.get_extractor(extract_fct="excerpt")
    assert isinstance(excerpt_extractor, ExcerptExtractor)

    flair_clause_extractor = ExtractorFactory.get_extractor(
        extract_fct="flair_clause", clause_type=['NP'], tagger=None)
    assert isinstance(flair_clause_extractor, FlairClauseExtractor)

    spacy_clause_extractor = ExtractorFactory.get_extractor(
        extract_fct="spacy_clause", clause_type=['NP'], pipeline=None)
    assert isinstance(spacy_clause_extractor, SpacyClauseExtractor)

    with pytest.raises(ValueError):
        ExtractorFactory.get_extractor(extract_fct="invalid")
