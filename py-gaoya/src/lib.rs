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
use crate::TokenizerOption::OnStack;
use crate::TokenizerOption::OnHeap;

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

/// OnStackTokenizer produces iterator of tokens as &str on stack.
/// Can be used only for basic tokenization
trait OnStackTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item=&'a str> + 'a>;
}

/// OnHeapTokenizer produces iterator of tokens as String on heap
trait OnHeapTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item=String>>;
}

struct WhiteSpaceTokenizer {}

impl OnStackTokenizer for WhiteSpaceTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item=&'a str> + 'a> {
        whitespace_split_boxed(text)
    }
}

struct CharShingletokenizer {
    range: (usize, usize),
}

impl OnStackTokenizer for CharShingletokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item=&'a str> + 'a> {
        shingle_text_boxed(text, self.range.0)
    }
}


struct WordShingleTokenizer {
    range: (usize, usize),
}

impl OnHeapTokenizer for WordShingleTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Box<dyn Iterator<Item=String>> {
        let words: Vec<_> = whitespace_split(text)
            .collect();

        let shingles: Shingles<[&str]> = Shingles::new(words.as_slice(), self.range.0);
        let shingles_owned: Vec<_> = shingles.into_iter()
            .map(|shingle| shingle.join(" "))
            .collect();
        Box::new(shingles_owned.into_iter())
    }
}


pub enum TokenizerOption {
    None,
    OnStack(Box<dyn OnStackTokenizer>),
    OnHeap(Box<dyn OnHeapTokenizer>)
}

unsafe impl Sync for TokenizerOption {}
unsafe impl Send for TokenizerOption {}


fn make_tokenizer(name: &str, range: Option<(usize, usize)>) -> TokenizerOption {
    if name == "char" {
        return match range {
            Some(range) => {
                return OnStack(Box::new(CharShingletokenizer { range: range}))
            },
            None => {
                println!("Invalid tokenizer {}", name);
                TokenizerOption::None
            }
        };
    } else if name == "word" {
        if range.is_some() {
            return OnHeap(Box::new(WordShingleTokenizer { range: range.unwrap()}));
        } else {
            return OnStack(Box::new(WhiteSpaceTokenizer{}));
        }
    }
    println!("Invalid tokenizer {}", name);
    TokenizerOption::None
}




