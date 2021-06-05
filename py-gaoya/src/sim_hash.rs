use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
extern crate gaoya;
use self::gaoya::simhash::SimSipHasher128;
use gaoya::simhash::{SimHash, SimHashIndex, SimSipHasher64};
use gaoya::text::tokenize_text;
use rayon::prelude::*;

#[pyclass]
struct SimHash64StringIntIndex {
    inner: gaoya::simhash::SimHashIndex<u64, u64>,
    sim_hash: SimHash<SimSipHasher64, u64, 64>,
}

#[pymethods]
impl SimHash64StringIntIndex {
    #[new]
    #[args(num_perm = "6s", max_distance = "5")]
    pub fn new(num_blocks: usize, max_distance: usize) -> PyResult<Self> {
        let index = SimHash64StringIntIndex {
            inner: SimHashIndex::new(num_blocks, max_distance),
            sim_hash: SimHash::<SimSipHasher64, u64, 64>::new(SimSipHasher64::new(5, 6)),
        };
        Ok(index)
    }

    pub fn insert_document(&mut self, id: u64, doc: String) {
        self.inner.insert(
            id,
            self.sim_hash.create_signature(tokenize_text(doc.as_str())),
        )
    }

    pub fn insert_tokens(&mut self, id: u64, tokens: Vec<&str>) {
        self.inner
            .insert(id, self.sim_hash.create_signature(tokens.iter()));
    }

    pub fn par_bulk_insert_tokens(&mut self, ids: Vec<u64>, docs_tokens: Vec<Vec<&str>>) {
        let signatures = docs_tokens
            .par_iter()
            .map(|tokens| self.sim_hash.create_signature(tokens.iter()))
            .collect();
        self.inner.park_bulk_insert(ids, signatures);
    }

    pub fn par_bulk_insert_docs(&mut self, ids: Vec<u64>, docs: Vec<&str>) {
        let signatures = docs
            .par_iter()
            .map(|doc| self.sim_hash.create_signature(tokenize_text(doc)))
            .collect();
        self.inner.park_bulk_insert(ids, signatures);
    }

    pub fn query(&self, doc: String) -> Vec<u64> {
        let signature = self.sim_hash.create_signature(tokenize_text(doc.as_str()));
        self.inner
            .query(&signature)
            .into_iter()
            .map(|id_ref| id_ref.clone())
            .collect()
    }

    pub fn query_tokens(&self, tokens: Vec<&str>) -> Vec<u64> {
        let signature = &self.sim_hash.create_signature(tokens.iter());
        self.inner
            .query(&signature)
            .into_iter()
            .map(|id_ref| id_ref.clone())
            .collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

#[pyclass]
struct SimHash128StringIntIndex {
    inner: gaoya::simhash::SimHashIndex<u128, u64>,
    sim_hash: SimHash<SimSipHasher128, u128, 128>,
}

#[pymethods]
impl SimHash128StringIntIndex {
    #[new]
    #[args(num_perm = "6s", max_distance = "5")]
    pub fn new(num_blocks: usize, max_distance: usize) -> PyResult<Self> {
        let index = SimHash128StringIntIndex {
            inner: SimHashIndex::new(num_blocks, max_distance),
            sim_hash: SimHash::<SimSipHasher128, u128, 128>::new(SimSipHasher128::new(5, 6)),
        };
        Ok(index)
    }

    pub fn insert_document(&mut self, id: u64, doc: String) {
        self.inner.insert(
            id,
            self.sim_hash.create_signature(tokenize_text(doc.as_str())),
        )
    }

    pub fn insert_tokens(&mut self, id: u64, tokens: Vec<&str>) {
        self.inner
            .insert(id, self.sim_hash.create_signature(tokens.iter()));
    }

    pub fn par_bulk_insert_tokens(&mut self, ids: Vec<u64>, docs_tokens: Vec<Vec<&str>>) {
        let signatures = docs_tokens
            .par_iter()
            .map(|tokens| self.sim_hash.create_signature(tokens.iter()))
            .collect();
        self.inner.park_bulk_insert(ids, signatures);
    }

    pub fn par_bulk_insert_docs(&mut self, ids: Vec<u64>, docs: Vec<&str>) {
        let signatures = docs
            .par_iter()
            .map(|doc| self.sim_hash.create_signature(tokenize_text(doc)))
            .collect();
        self.inner.park_bulk_insert(ids, signatures);
    }

    pub fn query(&self, doc: String) -> Vec<u64> {
        let signature = self.sim_hash.create_signature(tokenize_text(doc.as_str()));
        self.inner
            .query(&signature)
            .into_iter()
            .map(|id_ref| id_ref.clone())
            .collect()
    }

    pub fn query_tokens(&self, tokens: Vec<&str>) -> Vec<u64> {
        let signature = &self.sim_hash.create_signature(tokens.iter());
        self.inner
            .query(&signature)
            .into_iter()
            .map(|id_ref| id_ref.clone())
            .collect()
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

pub fn init_simhash_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<SimHash64StringIntIndex>()?;
    m.add_class::<SimHash128StringIntIndex>()?;
    Ok(())
}
