use pyo3::prelude::*;
use pyo3::{PyClass, PyObjectProtocol, PyTypeInfo};

use crate::TokenizerSpecification;
use fnv::FnvBuildHasher;
use gaoya::minhash::{compute_jaccard_similarity, MinHash, MinHash16V1, MinHash32V1,  MinHash32V2, MinHash64V1};
use gaoya::text::{shingle_text, shingle_text_range, whitespace_split};
use pyo3::class::impl_::PyClassImpl;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

extern crate gaoya;

macro_rules! py_minhash_index {
    ($name: ident, $type: ident, $minhash: ident, $hasher: ident) => {
        #[pyclass(unsendable)]
        pub struct $name {
            pub inner: gaoya::minhash::MinHashIndex<$type, i64>,
            pub min_hash: $minhash<$hasher>,
            pub tokenizer: TokenizerSpecification,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[args(
                threshold = "0.5",
                num_perm = "128",
                fpw = "0.5",
                fnw = "0.5",
                analyzer = "\"word\"",
                ngram_range = "(1,1)"
            )]
            pub fn new(
                threshold: f64,
                num_hashes: usize,
                fpw: f64,
                fnw: f64,
                analyzer: &str,
                ngram_range: (usize, usize),
            ) -> PyResult<Self> {
                let option_range = if ngram_range.0 == 1 && ngram_range.1 == 1 {
                    None
                } else {
                    Some(ngram_range)
                };
                let index = $name {
                    inner: gaoya::minhash::MinHashIndex::new_with_weights(
                        threshold, num_hashes, fpw, fnw,
                    ),
                    min_hash: $minhash::new(num_hashes),
                    tokenizer: TokenizerSpecification::new(analyzer, option_range),
                };
                Ok(index)
            }

            pub fn tokenize_and_minhash(&self, doc: &str) -> Vec<$type> {
                match &self.tokenizer {
                    TokenizerSpecification::CharShingle((from, Some(to))) => self
                        .min_hash
                        .create_signature(shingle_text_range(doc, *from, *to)),
                    TokenizerSpecification::CharShingle((from, None)) => {
                        self.min_hash.create_signature(shingle_text(doc, *from))
                    }
                    TokenizerSpecification::WhiteSpace() => {
                        self.min_hash.create_signature(whitespace_split(doc))
                    }
                }
            }

            pub fn insert_document(&mut self, id: i64, doc: String) {
                self.inner
                    .insert(id, self.tokenize_and_minhash(doc.as_str()))
            }

            pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
                self.inner
                    .insert(id, self.min_hash.create_signature(tokens.iter()));
            }

            pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
                let hashes: Vec<_> = docs
                    .par_iter()
                    .map(|doc| self.tokenize_and_minhash(doc))
                    .collect();
                self.inner.par_bulk_insert(ids, hashes);
            }

            pub fn bulk_insert_tokens(&mut self, ids: Vec<i64>, tokens: Vec<Vec<&str>>) {
                for id_tokens in ids.iter().zip(tokens.iter()) {
                    self.inner.insert(
                        *id_tokens.0,
                        self.min_hash.create_signature(id_tokens.1.iter()),
                    );
                }
            }

            pub fn par_bulk_insert_tokens(&mut self, ids: Vec<i64>, tokens: Vec<Vec<&str>>) {
                let hashes = self.min_hash.bulk_create_signature(&tokens);
                self.inner.par_bulk_insert(ids, hashes);
            }

            pub fn query_tokens(&self, tokens: Vec<&str>) -> Vec<i64> {
                let signature = &self.min_hash.create_signature(tokens.iter());
                self.inner.query_owned(signature).into_iter().collect()
            }

            pub fn query(&self, doc: String) -> Vec<i64> {
                self.inner
                    .query_owned(&self.tokenize_and_minhash(doc.as_str()))
                    .into_iter()
                    .collect()
            }

            pub fn size(&self) -> usize {
                self.inner.size()
            }
        }

        #[pyproto]
        impl PyObjectProtocol for $name {
            fn __str__(&self) -> PyResult<String> {
                Ok(format!("{}", self.inner))
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{}", self.inner))
            }
        }

    };
}

py_minhash_index!(MinHash64StringIntIndex, u64, MinHash64V1, FnvBuildHasher);
py_minhash_index!(MinHash32StringIntIndex, u32, MinHash32V1, FnvBuildHasher);
py_minhash_index!(MinHash16StringIntIndex, u16, MinHash16V1, FnvBuildHasher);


pub fn init_minhash_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<MinHash64StringIntIndex>()?;
    m.add_class::<MinHash32StringIntIndex>()?;
    m.add_class::<MinHash16StringIntIndex>()?;
    Ok(())
}
