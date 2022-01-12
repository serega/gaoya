import gaoya


def test_minhash_64bit():
    index = gaoya.minhash.MinHash64StringIntIndex(jaccard_threshold=0.5, num_bands=20, band_width=5, analyzer="word")
    index.insert_tokens(1, ['a', 'b', 'c', 'd', 'e'])

    assert index.query_tokens(['a', 'b', 'c', 'd', 'e']) == [1]
    assert index.query_tokens(['a', 'b', 'c', 'd', 'f']) == [1]
    assert index.query_tokens(['a', 'b', 'h', 'q', 'f']) == []

    assert index.query("a b c d e") == [1]


def test_minhash_32bit():
    index = gaoya.minhash.MinHash32StringIntIndex(0.5, 30, 5)
    index.insert_document(1, "a b c d e f")
    index.insert_document(2, "1 2 3 4 5 6 8")

    assert index.query_tokens(['a', 'b', 'c', 'd', 'e', 'f']) == [1]
    assert index.query_tokens(['a', 'b', 'c', 'd', 'e', 'g']) == [1]
    assert index.query_tokens(['a', 'b', 'h', 'q', 'f']) == []

    assert index.query("1 2 3 4 5 6 8") == [2]
    assert index.query("1 2 3 4 5 6 9 10") == [2]
    assert index.query_tokens(['9', '2', '3', '4', '5', '6', '7', '8']) == [2]

def test_minhash_16bit():
    index = gaoya.minhash.MinHash16StringIntIndex(0.5, 40, 5)
    index.insert_tokens(1, ['a', 'b', 'c', 'd', 'e'])

    assert index.query_tokens(['a', 'b', 'c', 'd', 'e']) == [1]
    assert index.query_tokens(['a', 'b', 'c', 'd', 'f']) == [1]
    assert index.query_tokens(['a', 'b', 'h', 'q', 'f']) == []



