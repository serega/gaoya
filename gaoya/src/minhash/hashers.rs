use core::convert::TryInto;
use seahash::SeaHasher;
use sha1::digest::Reset;
use sha1::{Digest, Sha1};
use siphasher::sip::SipHasher;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::str::FromStr;

#[derive(Clone, Debug)]
pub enum Hashers {
    Sip,
    Sha1,
    Sea,
}

impl Hashers {
    pub fn new_hasher(&self) -> Box<dyn Hasher> {
        match self {
            Hashers::Sha1 => Box::new(Sha1Hasher::new()),
            Hashers::Sip => Box::new(SipHasher::new_with_keys(1, 2)),
            Hashers::Sea => Box::new(SeaHasher::new()),
        }
    }

    pub fn from_str(input: &str) -> Result<Hashers, String> {
        match input.to_lowercase().as_str() {
            "sip" => Ok(Hashers::Sip),
            "sha1" => Ok(Hashers::Sha1),
            _ => Err(format!(
                "Unsupported hasher [{}]. Supported hashers [sip, sha1].",
                input
            )),
        }
    }
}

impl fmt::Display for Hashers {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Sha1Hasher {
    bytes: Vec<u8>,
}

impl Sha1Hasher {
    pub fn new() -> Self {
        Sha1Hasher { bytes: Vec::new() }
    }
}

impl Hasher for Sha1Hasher {
    fn finish(&self) -> u64 {
        let mut sha = Sha1::new();
        sha.write(self.bytes.as_slice());
        let result = sha.finalize();
        let hash64 = u64::from_be_bytes(result[0..8].try_into().unwrap());
        hash64
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
}
