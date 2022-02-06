from typing import List, Union, Tuple

from .gaoya import simhash as s

class SimHashStringIndex:
    """
    `SimHashStringIndex` for indexing and searching text documents based on approximation of cosine distance

    Reference: `Detecting Near-Duplicates for Web Crawling
    <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33026.pdf/>`_.

    Parameters
    ----------
    hash_size: int, default=64
        The size of simhash signature. Currently, only 64 and 128 bit hashes are supported

    num_blocks: int=6
        The number of blocks used in the index.

    hamming_distance:
        The  hamming distance threshold. The query method will return entries
        with hamming distance less than or equal to ``amming_distance`.

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
    """

    def __init__(self,
                 hash_size: int=64,
                 num_blocks: int=6,
                 hamming_distance: int=5,
                 analyzer: str='word',
                 lowercase: bool=False,
                 ngram_range: Tuple=None):

        if hash_size not in [64, 128]:
            raise ValueError(f"Invalid hash_size {hash_size}. hash_size must be one of of 64 or 128")
        if num_blocks < 3 and num_blocks > 32:
            raise ValueError(f"Invalid num_blocks {num_blocks}. num_blocks must be between 3 and 32")
        if hamming_distance >= num_blocks:
            raise ValueError(f"Number of blocks must be greater than hamming distance")
        if not callable(analyzer) and analyzer not in { 'word', 'char' }:
            raise ValueError(f"Analyzer must be word, char or callable")

        self.analyzer = analyzer
        # if analyzer is callable we need to pass something to index's constructor.
        analyzer = 'word' if callable(self.analyzer) else analyzer
        if hash_size == 64:
            self.index = s.SimHash64StringIntIndex(num_blocks, hamming_distance, analyzer, lowercase, ngram_range)
        else:
            self.index = s.SimHash128StringIntIndex(num_blocks, hamming_distance, analyzer, lowercase, ngram_range)


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

    def query(self, doc: str, return_hamming_distance=False) -> Union[List[int], List[Tuple[int, int]]]:
        """
        Searches the index for documents similar `doc`
        Returns list of similar document ids.
        If return_similarity is `True` method returns a list of tuples where the first element
        is document id and the second is hamming distance. The result is sorted by hamming distance
        from lowest to highest

        Parameters
        ----------
        doc: str
        return_similarity: Bool, default=False
            Whether to return hamming distance

        Returns:
        ----------
        List of ids or list of tuples
        """
        if callable(self.analyzer):
            if return_hamming_distance:
                return self.index.query_tokens_return_similarity(self.analyzer(doc))
            else:
                return self.index.query_tokens(self.analyzer(doc))
        else:
            if return_hamming_distance:
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
                return self.index.par_bulk_query_tokens_return_distance(analyzed_docs)
            else:
                return self.index.par_bulk_query_tokens(analyzed_docs)
        else:
            if return_similarity:
                return self.index.par_bulk_query_tokens_return_distance(docs)
            else:
                return self.index.par_bulk_query(docs)
