from typing import List, Union, Tuple

from .gaoya import minhash as m

class MinHashStringIndex:
    """
    MinHashStringIndex for indexing and searching text documents based jaccard similarity.


    Reference: `Chapter 3, Mining of Massive Datasets <http://www.mmds.org/>`
    If  `hash_size` and `num_bands` is specified `num_hashes` is not used, otherwise
        `hash_size` and `num_bands` will be calculated according to S-curve.

    Parameters
    ----------
    hash_size: int, default=32
        The size of individual hashes in bits in minhash signature. Supported sizes are (16, 32, 64).
        Bigger hashes offer better accuracy, smaller hashes use less memory.

    num_bands: int, default=25
        The number of bands

    band_size: int, default=5
        The number of hashes in individual band .
        The signature length is `num_bands` * `band_size`
        The signature length in bytes is
        (`num_bands` * `band_size` * `hash_size`) / 8

    num_hashes: int, default=None.
        The number of hashes in the signature. The argument is not used when
        `num_bands` and `band_size` are provided.


    jaccard_threshold: float, default=0.75.
        The jaccard similarity threshold. The query method will return documents that
        have jaccard similarity threshold greater than ``jaccard_threshold``.

    analyzer : {'word', 'char' } or callable, default='word'
        To create MinHash signature document must be tokenized into smaller units (features).
        Whether the feature should be made of word or character n-grams.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input. Note, that built-in analyzers are implemented in Rust,
        and generally faster that similar implementation in Python.

    lowercase : bool, default=False
        Convert all characters to lowercase before tokenizing.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an `ngram_range` of `(1, 1)`  means only
        unigrams, `(1, 2)` means unigrams and bigrams, and `(2, 2)` means
        only bigrams.
        Only applies if `analyzer` is not callable.

    Examples
    --------
            >>> index = gaoya.minhash.MinHashStringIndex(32, 0.5, 42, 3, None, 'word', True, (1,1))
            >>> corpus = [
            ...     'This is the first document.',
            ...     'This document is the second document.',
            ...     'And this is the third document.',
            ...     'Is this the first document?',
            ...     'This not the first nor the second nor the third, but the fourth document'
            ... ]
            >>> for i, doc in enumerate(corpus): index.insert_document(i, doc)
            >>> for i, doc in enumerate(corpus):
            ...     if i < 4:
            ...         assert set(index.query(doc)) == {0, 1, 2, 3}, str(index.query(doc))
            ...      else:
            ...         assert set(index.query(doc)) == {4}, str(index.query(doc))
            >>>
        """



    def __init__(self, hash_size=32,
                 jaccard_threshold=0.75,
                 num_bands=20,
                 band_size=5,
                 num_hashes=None,
                 analyzer='word',
                 lowercase=False,
                 ngram_range=None):
        if hash_size not in [8, 16, 32, 64]:
            raise ValueError(f"Invalid hash_size {hash_size}. hash_size must be on of 8, 16, 32 or 64")
        if jaccard_threshold < 0.0 or jaccard_threshold > 1.0:
            raise ValueError(f"Jaccard threshold must be between 0 and 1")
        self.analyzer = analyzer
        # if analyzer is callable we need to pass something to index's constructor.
        analyzer = 'word' if callable(self.analyzer) else analyzer
        if hash_size == 64:
            self.index = m.MinHash64StringIntIndex(jaccard_threshold, num_bands, band_size, num_hashes, analyzer, lowercase, ngram_range)
        elif hash_size == 32:
            self.index = m.MinHash32StringIntIndex(jaccard_threshold, num_bands, band_size, num_hashes, analyzer, lowercase, ngram_range)
        elif hash_size == 16:
            self.index = m.MinHash16StringIntIndex(jaccard_threshold, num_bands, band_size, num_hashes, analyzer, lowercase, ngram_range)
        elif hash_size == 8:
            self.index = m.MinHash8StringIntIndex(jaccard_threshold, num_bands, band_size, num_hashes, analyzer, lowercase, ngram_range)
        else:
            raise ValueError(f"Invalid hash size {hash_size}")


    def insert_document(self, id, doc):
        """
        Inserts a document `doc` with id `id` into the index.

        Parameters
        ----------

        id: int
            Id of the document
        doc: str
            Document text
        """
        if callable(self.analyzer):
            self.index.insert_tokens(id, self.analyzer(doc))
        else:
            self.index.insert_document(id, doc)

    def query(self, doc: str, return_similarity=False) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Searches the index for documents similar to `doc`.
        Returns list of similar document ids.
        If return_similarity is `True` method returns a list of tuples where the first element
        is document id and the second is jaccard similarity. The result is sorted by similarity from
        highest to lowest.

        Parameters
        ----------
        doc: str
        return_similarity: Bool, default=False
            Whether to return jaccard similarity values

        Returns:
        ----------
        List of ids or list of tuples
        """
        if callable(self.analyzer):
            if return_similarity:
                return self.index.query_tokens_return_similarity(self.analyzer(doc))
            else:
                return self.index.query_tokens(self.analyzer(doc))
        else:
            if return_similarity:
                return self.index.query_return_similarity(doc)
            else:
                return self.index.query(doc)

    def par_bulk_query(self, docs: List[str], return_similarity=False):
        """
        Searches the index for documents similar to `docs`.
        This method uses multiple native threads to execute `query` operation on a batch of documents
        Returns list of lists of similar document ids or list of lists of tuples
        Parameters
        ----------
        doc: list
            List of strings
        return_similarity: Bool, default=False
            Whether to return jaccard similarity values

        Returns:
        ----------
        List Lists of ids or list of lists of tuples
        """

        if callable(self.analyzer):
            analyzed_docs = [self.analyzer(doc) for doc in docs]
            if return_similarity:
                return self.index.par_bulk_query_tokens_return_similarity(analyzed_docs)
            else:
                return self.index.par_bulk_query_tokens(analyzed_docs)
        else:
            if return_similarity:
                return self.index.par_bulk_query_return_similarity(docs)
            else:
                return self.index.par_bulk_query(docs)



    def par_bulk_insert_docs(self, ids: List[int], docs: List[str]):
        """
        Inserts a batch of documents. This method will use multiple cores to insert a batch
        of documents into the index. If analyzer is callable tokenization will be single threaded.

        Parameters
        ----------
        ids: list
            List of ids

        docs: list
            List of strings
        """
        if callable(self.analyzer):
            tokens = [self.analyzer(doc) for doc in docs]
            self.index.bulk_insert_tokens(ids, tokens)
        else:
            self.index.par_bulk_insert_docs(ids, docs)

    def remove(self, id: int):
        """
        Removes id from the index.

        Currently, this method may not remove the minhash associated with `id` from memory if there is another minhash
        with the same value at any band.
        To fully remove minhash from memory gaoya needs to be compiled on nightly Rust channel with `--features "unstable"`
        This will work on the stable when this issue is resolved
        https://github.com/rust-lang/rust/issues/56167


        Parameters
        ----------
        id: int
            Id of the document
        """
        self.index.remove(id)

    def size(self):
        """
        Returns the number of documents in the index
        """
        return self.index.size()

    def __str__(self):
        return self.index.__str__()

    def __repr__(self):
        return self.index.__repr__()
