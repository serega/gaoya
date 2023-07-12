use pyo3::prelude::*;
extern crate gaoya;
use self::gaoya::simhash::SimSipHasher128;
use gaoya::simhash::{SimHash, SimHashIndex, SimSipHasher64};
use gaoya::text::{shingle_text,  shingle_text_range, whitespace_split, MultiShingles};
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
            #[pyo3(signature = (
                num_blocks = 6,
                max_distance = 5,
                analyzer = "word",
                lowercase = false,
                ngram_range = (1,1)
            ))]
            pub fn new(num_blocks: usize,
                       max_distance: usize,
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

            pub fn tokens2signature(&self, tokens: Vec<&str>) -> $type {
                self.sim_hash.create_signature(tokens.iter())
            }

            pub fn par_bulk_tokens2signatures(&self, docs_tokens: Vec<Vec<&str>>) -> Vec<$type> {
                docs_tokens.par_iter()
                    .map(|tokens| self.tokens2signature(tokens.to_vec()))
                    .collect()
            }

            fn doc2signature(&self, doc: &str) -> $type {
                if self.lowercase {
                    let doc = doc.to_lowercase();
                        self.tokenize_and_simhash(doc.as_str())
                    } else {
                    self.tokenize_and_simhash(doc)
                }
            }

            fn par_bulk_doc2signatures(&self, docs: Vec<&str>) -> Vec<$type> {
                docs.par_iter()
                    .map(|doc| self.doc2signature(&doc))
                    .collect()
            }



            pub fn insert_document(&mut self, id: i64, doc: &str) {
                let signature = self.doc2signature(doc);
                self.inner.insert(id, signature);
            }

            pub fn insert_tokens(&mut self, id: i64, tokens: Vec<&str>) {
                self.inner
                    .insert(id, self.sim_hash.create_signature(tokens.iter()));
            }

            pub fn insert_sig(&mut self, id: i64, signature: $type) {
                self.inner.insert(id, signature);
            }

            pub fn par_bulk_insert_sig_pairs(&mut self, id_signature_pairs: Vec<(i64, $type)>) {
                self.inner.par_bulk_insert_pairs(id_signature_pairs);
            }

            pub fn par_bulk_insert_tokens(&mut self, ids: Vec<i64>, docs_tokens: Vec<Vec<&str>>) {
                let signatures = docs_tokens
                    .par_iter()
                    .map(|tokens| self.sim_hash.create_signature(tokens.iter()))
                    .collect();
                self.inner.par_bulk_insert(ids, signatures);
            }

                // call map use self.insert_tokens
            pub fn par_bulk_insert_tokens_pairs(&mut self, docs_id_tokens: Vec<(i64, Vec<&str>)>) {
                // docs_id_tokens.par_iter()
                //     .map(|(id, tokens)|  self.inner.insert(*id, self.sim_hash.create_signature(tokens.iter())) )
                //     .collect()
                if docs_id_tokens.len() < 100 {
                    for (id, tokens) in docs_id_tokens.iter() {
                        self.insert_tokens(*id, tokens.to_vec())
                    }
                } else {
                    let id_signatures = docs_id_tokens
                        .par_iter()
                        .map(|(id, tokens)| (*id, self.tokens2signature(tokens.to_vec())))
                        .collect();
                    self.par_bulk_insert_sig_pairs(id_signatures);
                }
            }


            pub fn par_bulk_insert_docs(&mut self, ids: Vec<i64>, docs: Vec<&str>) {
                if ids.len() < 100 {
                    for (id, doc) in ids.iter().zip(docs.iter()) {
                        self.insert_document(*id, doc)
                    }
                } else {
                    let signatures = docs
                        .par_iter()
                        .map(|doc| self.doc2signature(doc))
                        .collect();
                    self.inner.par_bulk_insert(ids, signatures);
                }
            }

            pub fn query(&self, doc: &str) -> Vec<i64> {
                let signature = self.doc2signature(doc);
                self.inner
                    .query(&signature)
                    .into_iter()
                    .map(|id_ref| id_ref.clone())
                    .collect()
            }

            pub fn query_return_distance(&self, doc: &str) -> Vec<(i64, usize)> {
                let signature = self.doc2signature(doc);
                self.inner.query_return_distance(&signature)
            }

            pub fn query_tokens(&self, tokens: Vec<&str>) -> Vec<i64> {
                let signature = self.sim_hash.create_signature(tokens.iter());
                self.inner
                    .query(&signature)
                    .into_iter()
                    .map(|id_ref| id_ref.clone())
                    .collect()
            }

            pub fn query_one(&mut self, signature: $type) -> Option<(i64, usize)> {
                self.inner.query_one(&signature).map(|(id_ref, distance)| (*id_ref, distance))
            }
            pub fn query_sig(&self, signature: $type) -> Vec<i64> {
                self.inner
                    .query(&signature)
                    .into_iter()
                    .map(|id_ref| id_ref.clone())
                    .collect()
            }
            pub fn query_sig_return_distance(&self, signature: $type) -> Vec<(i64, usize)> {
                self.inner.query_return_distance(&signature)
            }

            pub fn query_tokens_return_distance(&self, tokens: Vec<&str>) -> Vec<(i64, usize)> {
                let signature = self.sim_hash.create_signature(tokens.iter());
                self.inner.query_return_distance(&signature)
            }

            pub fn par_bulk_query(&self, docs: Vec<&str>) -> Vec<Vec<i64>> {
                let signatures = self.par_bulk_doc2signatures(docs);
                self.inner.par_bulk_query(&signatures)
                    .into_iter()
                    .map(|set| set.into_iter().collect())
                    .collect()
            }

            pub fn par_bulk_query_return_distance(&self, docs: Vec<&str>) -> Vec<Vec<(i64, usize)>> {
                let signatures = self.par_bulk_doc2signatures(docs);
                self.inner.par_bulk_query_return_distance(&signatures)
            }

            pub fn par_bulk_query_tokens_return_similarity(&self, doc_tokens: Vec<Vec<&str>>) -> Vec<Vec<(i64, usize)>> {
                let signatures = doc_tokens.par_iter()
                    .map(|tokens| self.sim_hash.create_signature(tokens.iter()))
                    .collect();
                self.inner.par_bulk_query_return_distance(&signatures)
            }

            pub fn par_bulk_query_sigs_return_similarity(&self, signatures: Vec<$type>) -> Vec<Vec<(i64, usize)>> {
                self.inner.par_bulk_query_return_distance(&signatures)
            }
            pub fn par_bulk_query_sigs(&self, signatures: Vec<$type>) -> Vec<Vec<i64>> {
                self.inner.par_bulk_query(&signatures)
                    .into_iter()
                    .map(|set| set.into_iter().collect())
                    .collect()
            }

            pub fn size(&self) -> usize {
                self.inner.size()
            }

            pub fn iter(&self) -> Vec<(i64, $type)> {
                self.inner.iter().map(|(id, sig)| (*id, *sig)).collect()
            }
            // pub fn inner(&self) -> &gaoya::simhash::SimHashIndex<$type, i64> {
            //     &self.inner
            // }


            // pub fn remove(&mut self, id: i64) {
            //     self.inner.remove(&id);
            // }

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
