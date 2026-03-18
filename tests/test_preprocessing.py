from risk_aware.preprocessing.tfidf import TfidfTextPreprocessor


def test_tfidf_preprocessor_transforms_texts() -> None:
    pre = TfidfTextPreprocessor(max_features=100, ngram_range=(1, 1))
    texts = ["credit card issue", "mortgage payment problem"]
    pre.fit(texts)
    matrix = pre.transform(texts)
    assert matrix.shape[0] == 2
