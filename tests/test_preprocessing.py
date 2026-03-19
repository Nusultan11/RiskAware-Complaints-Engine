import numpy as np

from risk_aware.preprocessing.neural import NeuralTextPreprocessor, neural_clean
from risk_aware.preprocessing.tfidf import TfidfTextPreprocessor


def test_tfidf_preprocessor_transforms_texts() -> None:
    pre = TfidfTextPreprocessor(max_features=100, ngram_range=(1, 1))
    texts = ["credit card issue", "mortgage payment problem"]
    pre.fit(texts)
    matrix = pre.transform(texts)
    assert matrix.shape[0] == 2


def test_neural_clean_keeps_anonymization_and_numbers_as_tokens() -> None:
    text = "My XXXX account 12345 was blocked!!!"
    cleaned = neural_clean(text)
    assert "<anon>" in cleaned
    assert "<num>" in cleaned
    assert "!" not in cleaned


def test_neural_preprocessor_builds_token_ids_and_mask() -> None:
    pre = NeuralTextPreprocessor(max_vocab_size=100, min_token_freq=1, max_length=6)
    pre.fit(["credit card issue", "loan payment issue"])
    token_ids = pre.transform(["credit issue", "unknown token"])
    mask = pre.build_attention_mask(token_ids)

    assert token_ids.shape == (2, 6)
    assert mask.shape == (2, 6)
    assert token_ids.dtype == np.int32
    assert mask.dtype == np.uint8
