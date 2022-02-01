from typing import Set

from gaoya.minhash import MinHashStringIndex


def test_minhash_64bit():
    index = MinHashStringIndex(64, jaccard_threshold=0.5, num_bands=20, band_size=5, analyzer="word")
    index.insert_document(1, " a b c d e")

    assert index.query("a b c d e") == [1]
    assert index.query("a b c d f") == [1]
    assert index.query("a b g h f") == []



def test_minhash_32bit():
    index = MinHashStringIndex(32, 0.5, 30, 5)
    index.insert_document(1, "a b c d e f")
    index.insert_document(2, "1 2 3 4 5 6 8")

    assert index.query(" a b c d e f") == [1]
    assert index.query(" a b c d e g") == [1]
    assert index.query("a b h g f") == []

    assert index.query("1 2 3 4 5 6 8") == [2]
    assert index.query("1 2 3 4 5 6 9 10") == [2]

def test_minhash_16bit_custom_analyzer():
    def split_and_uppercase(doc):
        return [token.upper() for token in doc.split(" ")]

    index = MinHashStringIndex(
        hash_size=16,
        jaccard_threshold=0.5,
        num_bands=40, band_size=5, num_hashes=None,
        analyzer=split_and_uppercase,
        lowercase=False)
    index.insert_document(1, "a b c d e f g")
    index.insert_document(2, "foo bar baz")

    assert index.query("A B C D E F G") == [1]
    assert index.query("a b c d e f g") == [1]
    assert index.query("FOO bar baz") == [2]
    assert index.query("FOO bar BAZ") == [2]


def test_documents():
    index = MinHashStringIndex(32, 0.5, 42, 3, None, 'word', True, (1,1))
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third document.',
        'Is this the first document?',
        'This not the first nor the second nor the third, but the fourth document'

    ]
    for i, doc in enumerate(corpus):
        index.insert_document(i, doc)

    for i, doc in enumerate(corpus):
        if i < 4:
            assert set(index.query(doc)) == {0, 1, 2, 3}, str(index.query(doc))
        else:
            assert set(index.query(doc)) == {4}, str(index.query(doc))

    index.remove(0)
    index.remove(4)

    for i, doc in enumerate(corpus):
        if i < 4:
            assert set(index.query(doc)) == {1, 2, 3}, str(index.query(doc))
        else:
            assert set(index.query(doc)) == set(), str(index.query(doc))



def test_return_similarity():
    def _jaccard(s1: Set, s2: Set): return len(s1 & s2) / len(s1 | s2)
    index = MinHashStringIndex(32, 0.5, 45, 3, None, 'word', False, (1,1))
    corpus = [
        "a b c d e f g h k l m n o p q",
        "a b c d e f g h k l m n o p",
        "a b c d e f g h k l m n o",
        "a b c d e f g h k l m n",
        "1 2 3 4 5 6 9 8 9 10",
        "1 2 3 4 5 6 9 8 9 10 11 12",
        "1 2 3 4 5 6 9 8 9 10 11",
        "1 2 3 4 5 6 9 8 9 10 11 12 13"
    ]

    def check_jaccard(doc1, doc2,  value):
        true_jaccard = _jaccard(set(doc1.split(" ")), set(doc2.split(" ")))
        assert abs(true_jaccard - value) < 0.1

    index.par_bulk_insert_docs(list(range(0, len(corpus))), corpus)
    assert index.size() == 8
    result = index.query(corpus[0], return_similarity=True)
    assert len(result) == 4

    assert result[0][1] > result[1][1]
    check_jaccard(corpus[0], corpus[result[0][0]], result[0][1])

    assert result[1][1] > result[2][1]
    check_jaccard(corpus[0], corpus[result[1][0]], result[1][1])

    assert result[2][1] > result[3][1]
    check_jaccard(corpus[0], corpus[result[2][0]], result[2][1])
    check_jaccard(corpus[0], corpus[result[3][0]], result[3][1])


    result = index.query(corpus[7], return_similarity=True)
    assert len(result) == 4

    assert result[0][1] > result[1][1]
    check_jaccard(corpus[7], corpus[result[0][0]], result[0][1])

    assert result[1][1] > result[2][1]
    check_jaccard(corpus[7], corpus[result[1][0]], result[1][1])

    assert result[2][1] > result[3][1]
    check_jaccard(corpus[7], corpus[result[2][0]], result[2][1])
    check_jaccard(corpus[7], corpus[result[3][0]], result[3][1])







