use pyo3::prelude::*;
use gaoya::text::{whitespace_split_boxed, whitespace_split};
use gaoya::text::shingle_text_boxed;
use gaoya::text::shingle_text;

mod min_hash;
mod sim_hash;

use min_hash::init_minhash_module;
use sim_hash::init_simhash_module;
use std::hash::Hash;
use std::iter::Map;
use shingles::Shingles;
use crate::TokenizerSpecification::{CharShingle, WhiteSpace};

#[pymodule]
fn gaoya(py: Python, module: &PyModule) -> PyResult<()> {
    let minhash_module = PyModule::new(py, "minhash")?;
    init_minhash_module(minhash_module)?;
    module.add_submodule(minhash_module)?;

    let simhash_module = PyModule::new(py, "simhash")?;
    init_simhash_module(simhash_module)?;
    module.add_submodule(simhash_module)?;
    Ok(())
}


pub enum TokenizerSpecification {
    CharShingle((usize, Option<usize>)),
    WhiteSpace(),
}

impl TokenizerSpecification {
    pub fn new(name: &str, range: Option<(usize, usize)>) -> Self {
        if name == "char" {
            let range = range.unwrap();
            if range.0 == range.1 {
                return CharShingle((range.0, None));
            } else {
                return CharShingle((range.0, Some(range.1)));
            }
        } else {
            return WhiteSpace();
        }
    }
}
