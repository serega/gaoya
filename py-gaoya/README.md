# Gaoya

## About
This project implements Locality Sensitive Hashing algorithms and data structures for indexing and querying text documents. 
The primary use cases
for Gaoya are deduplication and clustering.

* MinHash | SimHash
* Powered by Rust
* Multi-threaded


```python
>>> import gaoya
>>> index = gaoya.minhash.MinHashStringIndex(hash_size=32, jaccard_threshold=0.5, num_bands=42, band_size=3, analyzer='word', lowercase=True, ngram_range=(1,1))
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third document.',
...     'Is this the first document?',
...     'This not the first nor the second nor the third, but the fourth document'
... ]
>>> 
>>> for i, doc in enumerate(corpus): index.insert_document(i, doc)
... 
>>> index.query('This is the first document.')
[0, 1, 2, 3]
>>> 
```

## Installation
```
$ pip3 install gaoya
```


## Examples
[Document Deduplication with Gaoya](https://github.com/serega/gaoya/blob/master/py-gaoya/examples/deduplication_scholarly_articles_gaoya.ipynb)

## References