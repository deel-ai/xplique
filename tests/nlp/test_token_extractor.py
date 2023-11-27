from xplique.commons.nlp import WordExtractor, SentenceExtractor
from xplique.commons.nlp import ClauseExtractor, ExcerptExtractor, ExtractorFactory

import pytest

@pytest.fixture
def example_sentence():
    return "One two three. Second sentence.Third Sentence, test1, test2; test3-test4 .GO!"\
           " Trust me,. sentence not starting with capital letter."\
           " Sentence with dots..Word, Word, word,...word ....so a sentence"

def test_word_extractor(example_sentence):
    extractor = WordExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '
    expected_tokens = [ 'One', 'two', 'three', '.',
                        'Second', 'sentence.Third', 'Sentence', ',', 'test1', ',', 'test2', ';',
                        'test3-test4', '.GO', '!', 'Trust', 'me', ',', '.', 'sentence', 'not',
                        'starting','with', 'capital', 'letter', '.', 'Sentence', 'with', 'dots',
                        '..', 'Word', ',', 'Word', ',', 'word', ',', '...', 'word', '....', 'so',
                        'a', 'sentence']
    assert tokens == expected_tokens, print('tokens:', tokens)

def test_word_extractor_ignore_words(example_sentence):
    extractor = WordExtractor(ignore_words = ['me', 'not', 'so', 'a',])
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '
    expected_tokens = [ 'One', 'two', 'three', '.',
                        'Second', 'sentence.Third', 'Sentence', ',', 'test1', ',', 'test2', ';',
                        'test3-test4', '.GO', '!', 'Trust', ',', '.', 'sentence', 'starting',
                        'with','capital', 'letter', '.', 'Sentence', 'with', 'dots', '..',
                        'Word', ',', 'Word', ',', 'word', ',', '...', 'word', '....', 'sentence']
    assert tokens == expected_tokens, print('tokens:', tokens)

def test_word_extractor_from_list(example_sentence):
    extractor = WordExtractor()
    tokens, separator = extractor.extract_tokens([example_sentence, example_sentence])
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == ' '

def test_sentence_extractor(example_sentence):
    extractor = SentenceExtractor()
    tokens, separator = extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    assert separator == '. '
    expected_tokens = [ 'One two three.',
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
    expected_tokens = [ 'One two three.',
                        'Second sentence.',
                        'Third Sentence, test1, test2; test3-test4 .',
                        'GO!',
                        'Trust me,.',
                        'Sentence with dots.',
                        'Word, Word, word,.']
    assert tokens == expected_tokens, print('tokens:', tokens)

def test_clause_extractor_close_type_none(example_sentence):
    clause_extractor = ClauseExtractor(clause_type = None)
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = [ 'One two three',
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


def test_clause_extractor_close_type_NP(example_sentence):
    clause_extractor = ClauseExtractor(clause_type = ['NP'])
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    expected_tokens = [ 'One two three',
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

def test_clause_extractor_close_type_ADJP(example_sentence):
    clause_extractor = ClauseExtractor(clause_type = ['ADJP'])
    tokens, separator = clause_extractor.extract_tokens(example_sentence)
    assert isinstance(tokens, list)
    assert isinstance(separator, str)
    print(tokens)
    expected_tokens = []
    assert tokens == expected_tokens

def test_extractor_factory():
    word_extractor = ExtractorFactory.get_extractor(extract_fct="word")
    assert isinstance(word_extractor, WordExtractor)

    sentence_extractor = ExtractorFactory.get_extractor(extract_fct="sentence")
    assert isinstance(sentence_extractor, SentenceExtractor)

    excerpt_extractor = ExtractorFactory.get_extractor(extract_fct="excerpt")
    assert isinstance(excerpt_extractor, ExcerptExtractor)

    clause_extractor = ExtractorFactory.get_extractor(extract_fct="clause", clause_type=['NP'])
    assert isinstance(clause_extractor, ClauseExtractor)

    with pytest.raises(ValueError):
        ExtractorFactory.get_extractor(extract_fct="invalid")
