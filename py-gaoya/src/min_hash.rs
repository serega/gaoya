use pyo3::prelude::*;


use crate::TokenizerSpecification;
use fnv::FnvBuildHasher;
use gaoya::minhash::{
    calculate_minhash_params,
    MinHasher, MinHasher8, MinHasher16, MinHasher32, MinHasher64V1,
    HashSetContainer, SmallVecContainer, VecContainer
};
use gaoya::text::{shingle_text, shingle_text_range, whitespace_split, MultiShingles};
use shingles::Shingles;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

extern crate gaoya;

macro_rules! py_minhash_index {
    ($name: ident, $type: ident, $stype: expr, $container_type: ident,  $minhash: ident, $hasher: ident) => {
        #[doc = concat!("MinHash", $stype, "StringIntIndex is a MinhashIndex that uses ", $stype, " hashes, ")]
        #[doc = "for string values and integer keys."]
        #[pyclass(unsendable)]
        pub struct $name {
            pub inner: gaoya::minhash::MinHashIndex<$type, i64, $container_type>,
            pub min_hash: $minhash<$hasher>,
            pub tokenizer: TokenizerSpecification,
            pub lowercase: bool,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[args(
                jaccard_threshold = "0.5",
                num_bands = "42",
                band_width = "3",
                num_hashes = "126",
                analyzer = "\"word\"",
                lowercase = "false",
                ngram_range = "(1,1)"
            )]
            pub fn new(
                jaccard_threshold: f64,
                num_bands: Option<usize>,
                band_width: Option<usize>,
                num_hashes: Option<usize>,
                analyzer: Option<&str>,
                lowercase: Option<bool>,
                ngram_range: Option<(usize, usize)>,
            ) -> PyResult<Self> {
                if let (Some(num_bands), Some(band_width)) = (num_bands, band_width) {
                    let index = $name {
                        inner: gaoya::minhash::MinHashIndex::<_, _, $container_type>::new_index(num_bands, band_width, jaccard_threshold),
                        min_hash: $minhash::new(num_bands * band_width),
                        tokenizer: TokenizerSpecification::new(analyzer.unwrap_or("word"), ngram_range),
                        lowercase: lowercase.unwrap_or(false),
                    };
                    return Ok(index);
                }
                if let Some(num_hashes) = num_hashes {
                    let (num_bands, band_width) = calculate_minhash_params(jaccard_threshold, num_hashes);
                    let index = $name {
                        inner: gaoya::minhash::MinHashIndex::<_,_, $container_type>::new_index(num_bands, band_width, jaccard_threshold),
                        min_hash: $minhash::new(num_bands * band_width),
                        tokenizer: TokenizerSpecification::new(analyzer.unwrap_or("word"), ngram_range),
                        lowercase: lowercase.unwrap_or(false),
                    };
                    return Ok(index);
                }
                return Err(PyValueError::new_err("Either (num_bands, band_width) or num_hashes must be specified") );
            }

            pub fn tokenize_and_minhash(&self, doc: &str) -> Vec<$type> {
                match &self.tokenizer {
                    TokenizerSpecification::CharShingle((from, None)) => {
                        self.min_hash.create_signature(shingle_text(doc, *from))
                    }
                    TokenizerSpecification::CharShingle((from, Some(to))) => self
                        .min_hash
                        .create_signature(shingle_text_range(doc, *from, *to)),
                    TokenizerSpecification::WhiteSpace() => {
                        self.min_hash.create_signature(whitespace_split(doc))
                    }
                    TokenizerSpecification::WhiteSpaceShingle((from, None)) => {
                        let words: Vec<_> = whitespace_split(doc).collect();
                        let shingles = Shingles::new(words.as_slice(), *from);
                        self.min_hash.create_signature(shingles)
                    }
                    TokenizerSpecification::WhiteSpaceShingle((from, Some(to))) => {
                        let words: Vec<_> = whitespace_split(doc).collect();
                        let shingles = MultiShingles::new(words.as_slice(), *from, *to);
                        self.min_hash.create_signature(shingles)
                    }
                }
            }

            pub fn insert_document(&mut self, id: i64, doc: &str) {
                if self.lowercase {
                    let doc = doc.to_lowercase();
                    self.inner
                        .insert(id, self.tokenize_and_minhash(doc.as_str()))
                } else {
                    self.inner
                        .insert(id, self.tokenize_and_minhash(doc))
                }
            }

            pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
                self.inner
                    .insert(id, self.min_hash.create_signature(tokens.iter()));
            }

            pub fn remove(&mut self, id: i64) {
                self.inner.remove(&id);
            }

            fn bulk_hash_docs(&self, docs: Vec<&str>) -> Vec<Vec<$type>> {
                docs.par_iter()
                .map(|doc| {
                    if self.lowercase {
                        let doc = doc.to_lowercase();
                        self.tokenize_and_minhash(doc.as_str())
                    } else {
                        self.tokenize_and_minhash(doc)
                    }
                })
                .collect()
            }

            pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
                if ids.len() < 100 { // TODO: find a reasonable threshold
                    for (id, doc) in ids.iter().zip(docs.iter()) {
                        self.insert_document(*id, doc)
                    }
                } else {
                    let signatures = self.bulk_hash_docs(docs);
                    self.inner.par_bulk_insert(ids, signatures);
                }
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

            pub fn query_tokens_return_similarity(&self, tokens: Vec<&str>) ->  Vec<(i64, f64)> {
                let signature = &self.min_hash.create_signature(tokens.iter());
                self.inner.query_owned_return_similarity(&signature)
            }

            pub fn query(&self, doc: &str) -> Vec<i64> {
                let signature = if self.lowercase {
                    let doc = doc.to_lowercase();
                    self.tokenize_and_minhash(doc.as_str())
                } else {
                    self.tokenize_and_minhash(doc)
                };
                self.inner
                    .query_owned(&signature)
                    .into_iter()
                    .collect()
            }

            pub fn query_return_similarity(&self, doc: &str) -> Vec<(i64, f64)> {
                let signature = if self.lowercase {
                    let doc = doc.to_lowercase();
                    self.tokenize_and_minhash(doc.as_str())
                } else {
                    self.tokenize_and_minhash(doc)
                };
                self.inner.query_owned_return_similarity(&signature)
            }

            pub fn par_bulk_query(&self, docs: Vec<&str>) -> Vec<Vec<i64>> {
                let signatures = self.bulk_hash_docs(docs);
                self.inner.par_bulk_query(&signatures)
                    .into_iter()
                    .map(|set| set.into_iter().collect())
                    .collect()
            }

            pub fn par_bulk_query_return_similarity(&self, docs: Vec<&str>) -> Vec<Vec<(i64, f64)>> {
                let signatures = self.bulk_hash_docs(docs);
                self.inner.par_bulk_query_return_similarity(&signatures)
            }


            pub fn par_bulk_query_tokens(&self, tokens: Vec<Vec<&str>>) -> Vec<Vec<i64>> {
                let signatures = self.min_hash.bulk_create_signature(&tokens);
                self.inner.par_bulk_query(&signatures)
                    .into_iter()
                    .map(|set| set.into_iter().collect())
                    .collect()
            }

            pub fn par_bulk_query_tokens_return_similarity(&self, tokens: Vec<Vec<&str>>) -> Vec<Vec<(i64, f64)>> {
                let signatures = self.min_hash.bulk_create_signature(&tokens);
                self.inner.par_bulk_query_return_similarity(&signatures)
            }


            pub fn size(&self) -> usize {
                self.inner.size()
            }
            fn __str__(&self) -> PyResult<String> {
                Ok(format!("{} {:?}", self.inner, self.tokenizer))
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{} {:?}", self.inner, self.tokenizer))
            }

        }


    };
}

type HashSetContaineri64 = HashSetContainer<i64>;
type VecContaineri64 = VecContainer<i64>;
type SmallVecContaineri64 = SmallVecContainer<i64, 2>;

py_minhash_index!(MinHash64StringIntIndexHashSet, u64, "64", HashSetContaineri64, MinHasher64V1, FnvBuildHasher);
py_minhash_index!(MinHash64StringIntIndexVec, u64, "64", VecContaineri64, MinHasher64V1, FnvBuildHasher);
py_minhash_index!(MinHash64StringIntIndexSmallVec, u64, "64", SmallVecContaineri64, MinHasher64V1, FnvBuildHasher);

py_minhash_index!(MinHash32StringIntIndexHashSet, u32, "32", HashSetContaineri64, MinHasher32, FnvBuildHasher);
py_minhash_index!(MinHash32StringIntIndexVec, u32, "32", VecContaineri64, MinHasher32, FnvBuildHasher);
py_minhash_index!(MinHash32StringIntIndexSmallVec, u32, "32", SmallVecContaineri64, MinHasher32, FnvBuildHasher);


py_minhash_index!(MinHash16StringIntIndexHashSet, u16, "16", HashSetContaineri64, MinHasher16, FnvBuildHasher);
py_minhash_index!(MinHash16StringIntIndexVec, u16, "16", VecContaineri64, MinHasher16, FnvBuildHasher);
py_minhash_index!(MinHash16StringIntIndexSmallVec, u16, "16", SmallVecContaineri64, MinHasher16, FnvBuildHasher);

py_minhash_index!(MinHash8StringIntIndexHashSet, u8, "8", HashSetContaineri64, MinHasher8, FnvBuildHasher);
py_minhash_index!(MinHash8StringIntIndexVec, u8, "8", VecContaineri64, MinHasher8, FnvBuildHasher);
py_minhash_index!(MinHash8StringIntIndexSmallVec, u8, "8", SmallVecContaineri64, MinHasher8, FnvBuildHasher);



pub fn init_minhash_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<MinHash64StringIntIndexHashSet>()?;
    m.add_class::<MinHash64StringIntIndexVec>()?;
    m.add_class::<MinHash64StringIntIndexSmallVec>()?;

    m.add_class::<MinHash32StringIntIndexHashSet>()?;
    m.add_class::<MinHash32StringIntIndexVec>()?;
    m.add_class::<MinHash32StringIntIndexSmallVec>()?;

    m.add_class::<MinHash16StringIntIndexHashSet>()?;
    m.add_class::<MinHash16StringIntIndexVec>()?;
    m.add_class::<MinHash16StringIntIndexSmallVec>()?;

    m.add_class::<MinHash8StringIntIndexHashSet>()?;
    m.add_class::<MinHash8StringIntIndexVec>()?;
    m.add_class::<MinHash8StringIntIndexSmallVec>()?;

    Ok(())
}
