# Gaoya

## About
This project implements Locality Sensitive Hashing algorithms and data structures for indexing and querying text documents.
The primary use cases
for Gaoya are deduplication and clustering.

* MinHash | SimHash
* Rust | Python
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


## Rust Example

```rust
    use gaoya::minhash::{MinHashIndex, MinHasher32V1, MinHasher} ;
    use gaoya::text::whitespace_split;

    let corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third document.",
        "Is this the first document?",
        "This not the first nor the second nor the third, but the fourth document"];
    let minhasher = MinHasher32V1::new(42 * 3);
    let mut index = MinHashIndex::new(42, 3, 0.5);
    for (i, doc) in corpus.iter().enumerate() {
        index.insert(i, minhasher.create_signature(whitespace_split(&doc.to_lowercase())));
    }
    for (i, doc) in corpus.iter().enumerate() {
        if i < 4 {
            let mut expected = FxHashSet::default();
            expected.extend(vec![0, 1, 2, 3].into_iter());
            assert_eq!(index.query_owned(&minhasher.create_signature(whitespace_split(&doc.to_lowercase()))), expected);
        } else {
            let mut expected = FxHashSet::default();
            expected.insert(4);
            assert_eq!(index.query_owned(&minhasher.create_signature(whitespace_split(&doc.to_lowercase()))), expected);
        }
    }

```
## References