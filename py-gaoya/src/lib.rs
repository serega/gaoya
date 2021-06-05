use pyo3::prelude::*;

mod min_hash;
mod sim_hash;

use min_hash::init_minhash_module;
use sim_hash::init_simhash_module;

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
