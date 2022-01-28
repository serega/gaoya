use pyo3::prelude::*;

mod min_hash;
mod sim_hash;

use min_hash::init_minhash_module;
use sim_hash::init_simhash_module;
use crate::TokenizerSpecification::{CharShingle, WhiteSpace, WhiteSpaceShingle};

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

#[derive(Debug)]
pub enum TokenizerSpecification {
    CharShingle((usize, Option<usize>)),
    WhiteSpace(),
    WhiteSpaceShingle((usize, Option<usize>)),
}

impl TokenizerSpecification {
    pub fn new(name: &str, range: Option<(usize, usize)>) -> Self {
        if name == "char" {
            match range {
                Some(range) => {
                    if range.0 == range.1 {
                        return CharShingle((range.0, None));
                    } else {
                        return CharShingle((range.0, Some(range.1)));
                    }
                }
                None => {
                    return CharShingle((3, Some(4)))
                }
            }
        } else {
            match range {
                Some(range) => {
                    if range.0 == range.1 {
                        return WhiteSpaceShingle((range.0, None));
                    } else {
                        return WhiteSpaceShingle((range.0, Some(range.1)));
                    }
                }
                None => {
                    return WhiteSpace();
                }
            }

        }
    }
}
