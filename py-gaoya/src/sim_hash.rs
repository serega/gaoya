use pyo3::prelude::*;
extern crate gaoya;
use self::gaoya::simhash::SimSipHasher128;
use gaoya::simhash::{SimHash, SimHashIndex, SimSipHasher64};
use gaoya::text::{shingle_text,  shingle_tokens, shingle_text_range, whitespace_split, MultiShingles};
use shingles::Shingles;
use rayon::prelude::*;
use crate::TokenizerSpecification;


macro_rules! py_simhash_index {
    ($name: ident, $type: ident, $bitlen: expr, $stype: expr, $simhasher: ident) => {
        #[pyclass(unsendable)]
        pub struct $name {
            inner: gaoya::simhash::SimHashIndex<$type, i64>,
            sim_hash: SimHash<$simhasher, $type, $bitlen>,
            tokenizer: TokenizerSpecification,
            pub lowercase: bool,
        }
        #[pymethods]
        impl $name {
        #[new]
        #[args(
            num_blocks = "6",
            max_distance = "5",
            analyzer = "\"word\"",
            lowercase = "false",
            ngram_range = "(1,1)"
        )]
        pub fn new(num_blocks: usize, max_distance: usize,
                   analyzer: Option<&str>,
                   lowercase: Option<bool>,
                   ngram_range: Option<(usize, usize)>) -> PyResult<Self> {
            let index = $name {
                inner: SimHashIndex::new(num_blocks, max_distance),
                sim_hash: SimHash::<$simhasher, $type, $bitlen>::new($simhasher::new(5, 6)),
                tokenizer: TokenizerSpecification::new(analyzer.unwrap_or("word"), ngram_range),
                lowercase: lowercase.unwrap_or(false),
            };
            Ok(index)
        }

        pub fn tokenize_and_simhash(&self, doc: &str) -> $type {
            match &self.tokenizer {
                TokenizerSpecification::CharShingle((from, Some(to))) => self
                    .sim_hash
                    .create_signature(shingle_text_range(doc, *from, *to)),
                TokenizerSpecification::CharShingle((from, None)) => {
                    self.sim_hash.create_signature(shingle_text(doc, *from))
                }
                TokenizerSpecification::WhiteSpace() => {
                    self.sim_hash.create_signature(whitespace_split(doc))
                }
                TokenizerSpecification::WhiteSpaceShingle((from, None)) => {
                    let words: Vec<_> = whitespace_split(doc).collect();
                    let shingles = Shingles::new(words.as_slice(), *from);
                    self.sim_hash.create_signature(shingles)
                }
                TokenizerSpecification::WhiteSpaceShingle((from, Some(to))) => {
                    let words: Vec<_> = whitespace_split(doc).collect();
                    let shingles = MultiShingles::new(words.as_slice(), *from, *to);
                    self.sim_hash.create_signature(shingles)
                }
            }
        }


        pub fn insert_document(&mut self, id: i64, doc: &str) {
            if self.lowercase {
                let doc = doc.to_lowercase();
                self.inner.insert(
                    id,
                    self.tokenize_and_simhash(doc.as_str()));
            } else {
                self.inner.insert(
                    id,
                    self.tokenize_and_simhash(doc));
            }
        }

        pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
            self.inner
                .insert(id, self.sim_hash.create_signature(tokens.iter()));
        }

        pub fn par_bulk_insert_tokens(&mut self, ids: Vec<i64>, docs_tokens: Vec<Vec<&str>>) {
            let signatures = docs_tokens
                .par_iter()
                .map(|tokens| self.sim_hash.create_signature(tokens.iter()))
                .collect();
            self.inner.park_bulk_insert(ids, signatures);
        }

        pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
            if ids.len() < 100 {
                for (id, doc) in ids.iter().zip(docs.iter()) {
                    self.insert_document(*id, doc)
                }
            } else {
                let signatures = docs
                    .par_iter()
                    .map(|doc| {
                        if self.lowercase {
                            let doc = doc.to_lowercase();
                            self.tokenize_and_simhash(doc.as_str())
                        } else {
                            self.tokenize_and_simhash(doc)
                        }
                    })
                    .collect();
                self.inner.park_bulk_insert(ids, signatures);
            }
        }

        pub fn query(&self, doc: &str) -> Vec<i64> {
            let signature = if self.lowercase {
                let doc = doc.to_lowercase();
                self.tokenize_and_simhash(doc.as_str())
            } else {
                self.tokenize_and_simhash(doc)
            };

            self.inner
                .query(&signature)
                .into_iter()
                .map(|id_ref| id_ref.clone())
                .collect()
        }


    }

    }
}

py_simhash_index!(SimHash64StringIntIndex, u64, 64, "64" , SimSipHasher64);
py_simhash_index!(SimHash128StringIntIndex, u128, 128, "128" , SimSipHasher128);

pub fn init_simhash_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<SimHash64StringIntIndex>()?;
    m.add_class::<SimHash128StringIntIndex>()?;
    Ok(())
}
