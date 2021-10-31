use pyo3::prelude::*;
use pyo3::{PyObjectProtocol, PyClass, PyTypeInfo};

use gaoya::minhash::{
    compute_jaccard_similarity, Hashers, MinHash16, MinHash16V1, MinHash32, MinHash32V2, MinHash64,
    MinHash64V1,
};
use gaoya::text::whitespace_split;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use pyo3::class::impl_::PyClassImpl;
use crate::{OnStackTokenizer, OnHeapTokenizer, TokenizerOption, make_tokenizer};

extern crate gaoya;

#[pyclass]
struct Minhash64StringIntIndex {
    inner: gaoya::minhash::MinHashIndex<u64, i64>,
    min_hash: MinHash64V1,
}

#[pymethods]
impl Minhash64StringIntIndex {
    #[new]
    #[args(
        threshold = "0.5",
        num_perm = "128",
        fpw = "0.5",
        fnw = "0.5",
        hashfunc = "\"sip\""
    )]
    pub fn new(
        threshold: f64,
        num_perm: usize,
        fpw: f64,
        fnw: f64,
        hashfunc: &str,
    ) -> PyResult<Self> {
        match Hashers::from_str(hashfunc) {
            Err(e) => Err(PyValueError::new_err(e)),
            Ok(hasher) => {
                let index = Minhash64StringIntIndex {
                    inner: gaoya::minhash::MinHashIndex::new_with_weights(
                        threshold, num_perm, fpw, fnw,
                    ),
                    min_hash: MinHash64V1::new_with_hasher(num_perm, hasher),
                };
                Ok(index)
            }
        }
    }

    pub fn insert_document(&mut self, id: i64, doc: String) {
        self.inner.insert(
            id,
            self.min_hash.create_signature(whitespace_split(doc.as_str())),
        );
    }

    pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
        self.inner
            .insert(id, self.min_hash.create_signature(tokens.iter()));
    }

    pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
        let docs_tokens = docs
            .par_iter()
            .map(|doc| whitespace_split(doc).collect())
            .collect();
        self.par_bulk_insert_tokens(ids, docs_tokens);
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

    pub fn query(&self, text: String) -> Vec<i64> {
        let signature = &self.min_hash.create_signature(whitespace_split(text.as_str()));
        self.inner.query_owned(signature).into_iter().collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }
}


#[pyproto]
impl PyObjectProtocol for Minhash64StringIntIndex {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

#[pyclass]
struct MinHash32StringIntIndex {
    inner: gaoya::minhash::MinHashIndex<u32, i64>,
    min_hash: MinHash32V2,
}

#[pymethods]
impl MinHash32StringIntIndex {
    #[new]
    #[args(
        threshold = "0.5",
        num_perm = "128",
        fpw = "0.5",
        fnw = "0.5",
        hashfunc = "\"sip\""
    )]
    pub fn new(
        threshold: f64,
        num_perm: usize,
        fpw: f64,
        fnw: f64,
        hashfunc: &str,
    ) -> PyResult<Self> {
        match Hashers::from_str(hashfunc) {
            Err(e) => Err(PyValueError::new_err(e)),
            Ok(hasher) => {
                let index = MinHash32StringIntIndex {
                    inner: gaoya::minhash::MinHashIndex::new_with_weights(
                        threshold, num_perm, fpw, fnw,
                    ),
                    min_hash: MinHash32V2::new_with_hasher(num_perm, hasher),
                };
                Ok(index)
            }
        }
    }

    pub fn insert_document(&mut self, id: i64, doc: String) {
        self.inner.insert(
            id,
            self.min_hash.create_signature(whitespace_split(doc.as_str())),
        );
    }

    pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
        self.inner
            .insert(id, self.min_hash.create_signature(tokens.iter()));
    }

    pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
        let docs_tokens = docs
            .par_iter()
            .map(|doc| whitespace_split(doc).collect())
            .collect();
        self.par_bulk_insert_tokens(ids, docs_tokens);
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

    pub fn query(&self, text: String) -> Vec<i64> {
        let signature = &self.min_hash.create_signature(whitespace_split(text.as_str()));
        self.inner.query_owned(signature).into_iter().collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

#[pyproto]
impl PyObjectProtocol for MinHash32StringIntIndex {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

#[pyclass(unsendable)]
struct MinHash16StringIntIndex {
    inner: gaoya::minhash::MinHashIndex<u16, i64>,
    min_hash: MinHash16V1,
    tokenizer: TokenizerOption
}

#[pymethods]
impl MinHash16StringIntIndex {
    #[new]
    #[args(
        threshold = "0.5",
        num_perm = "128",
        fpw = "0.5",
        fnw = "0.5",
        hashfunc = "\"sip\"",
        analyzer = "\"word\"",
        ngram_range = "(1,1)",
    )]
    pub fn new(
        threshold: f64,
        num_perm: usize,
        fpw: f64,
        fnw: f64,
        hashfunc: &str,
        analyzer: &str,
        ngram_range: (usize, usize)
    ) -> PyResult<Self> {
        let option_range = if ngram_range.0 == 1 && ngram_range.1 == 1 { Some(ngram_range) } else { None };
        match Hashers::from_str(hashfunc) {
            Err(e) => Err(PyValueError::new_err(e)),
            Ok(hasher) => {
                let index = MinHash16StringIntIndex {
                    inner: gaoya::minhash::MinHashIndex::new_with_weights(
                        threshold, num_perm, fpw, fnw,
                    ),
                    min_hash: MinHash16V1::new_with_hasher(num_perm, hasher),
                    tokenizer: make_tokenizer(analyzer, option_range)
                };
                Ok(index)
            }
        }
    }

    pub fn insert_document(&mut self, id: i64, doc: String) {
        match &self.tokenizer {
            TokenizerOption::OnStack(tokenizer) => {
                self.inner.insert(
                    id,
                    self.min_hash.create_signature(tokenizer.tokenize(doc.as_str())),
                );
            },
            TokenizerOption::OnHeap(tokenizer) => {
                self.inner.insert(
                    id,
                    self.min_hash.create_signature(tokenizer.tokenize(doc.as_str())),
                );
            },
            TokenizerOption::None => ()
        }

    }

    pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
        self.inner
            .insert(id, self.min_hash.create_signature(tokens.iter()));
    }

    pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
        let docs_tokens = docs
            .par_iter()
            .map(|doc| whitespace_split(doc).collect())
            .collect();
        self.par_bulk_insert_tokens(ids, docs_tokens);
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

    pub fn query(&self, text: String) -> Vec<i64> {
        let signature = &self.min_hash.create_signature(whitespace_split(text.as_str()));
        self.inner.query_owned(signature).into_iter().collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

#[pyproto]
impl PyObjectProtocol for MinHash16StringIntIndex {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

/*

#[pyclass]
struct MinHash32IntIntIndex {
    inner: gaoya::minhash::MinHashIndex<u32, i32>,
    min_hash: MinHash32V2,
}

#[pymethods]
impl MinHash32IntIntIndex {
    #[new]
    #[args(
        threshold = "0.5",
        num_perm = "128",
        fpw = "0.5",
        fnw = "0.5",
        hashfunc = "\"sip\""
    )]
    pub fn new(
        threshold: f64,
        num_perm: usize,
        fpw: f64,
        fnw: f64,
        hashfunc: &str,
    ) -> PyResult<Self> {
        match Hashers::from_str(hashfunc) {
            Err(e) => Err(PyValueError::new_err(e)),
            Ok(hasher) => {
                let index = MinHash32IntIntIndex {
                    inner: gaoya::minhash::MinHashIndex::new_with_weights(
                        threshold, num_perm, fpw, fnw,
                    ),
                    min_hash: MinHash32V2::new_with_hasher(num_perm, hasher),
                };
                Ok(index)
            }
        }
    }

    #[new]
    pub fn new_with_params(num_perm: usize, b: usize, r: usize, hashfunc: &str) -> PyResult<Self> {
        match Hashers::from_str(hashfunc) {
            Err(e) => Err(PyValueError::new_err(e)),
            Ok(hasher) => {
                let index = MinHash32IntIntIndex {
                    inner: gaoya::minhash::MinHashIndex::new_with_params(b, r),
                    min_hash: MinHash32V2::new_with_hasher(num_perm, hasher),
                };
                Ok(index)
            }
        }
    }

    pub fn insert(&mut self, id: i32, value: Vec<i32>) {
        self.inner
            .insert(id, self.min_hash.create_signature(value.iter()));
    }

    pub fn bulk_insert(&mut self, ids: Vec<i32>, tokens: Vec<Vec<i32>>) {
        for id_tokens in ids.iter().zip(tokens.iter()) {
            self.inner.insert(
                *id_tokens.0,
                self.min_hash.create_signature(id_tokens.1.iter()),
            );
        }
    }

    pub fn par_bulk_insert(&mut self, ids: Vec<i32>, tokens: Vec<Vec<i32>>) {
        let hashes = self.min_hash.bulk_create_signature(&tokens);
        self.inner.par_bulk_insert(ids, hashes);
    }

    pub fn query(&self, value: Vec<i32>) -> Vec<i32> {
        let min_hashes = &self.min_hash.create_signature(value.iter());
        self.inner
            .query(min_hashes)
            .into_iter()
            .map(|id| id.to_owned())
            .collect()
    }

    pub fn query_top_k(&self, value: Vec<i32>, k: usize) -> Vec<i32> {
        let min_hashes = &self.min_hash.create_signature(value.iter());
        self.inner
            .query_top_k(min_hashes, k)
            .into_iter()
            .map(|k| k.0)
            .collect()
    }

    pub fn query_top_ks(&self, queries: Vec<Vec<i32>>, k: usize) -> Vec<Vec<i32>> {
        queries
            .into_par_iter()
            .map(|query| self.query_top_k(query, k.clone()))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }

    pub fn bulk_compute_jaccard_similarity(
        &self,
        query: Vec<i32>,
        sets: Vec<Vec<i32>>,
    ) -> Vec<f32> {
        sets.par_iter()
            .map(|set| compute_jaccard_similarity(query.iter(), set.iter()))
            .collect()
    }
}



#[pyproto]
impl PyObjectProtocol for MinHash32IntIntIndex {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

 */

pub fn init_minhash_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<Minhash64StringIntIndex>()?;
    m.add_class::<MinHash32StringIntIndex>()?;
    m.add_class::<MinHash16StringIntIndex>()?;
    //m.add_class::<MinHash32IntIntIndex>()?;
    Ok(())
}
