use siphasher::sip::SipHasher;
use siphasher::sip128::{Hasher128, SipHasher as SipHasher128};
use std::hash::{Hash, Hasher};
use crate::minhash::Sha1Hasher;
use xxhash_rust::xxh3::Xxh3;

pub trait SimHasher: Sized {
    type T;
    fn hash<U>(&self, item: &U) -> Self::T
    where
        Self: Sized,
        U: Hash;
}

pub struct SimSipHasher64 {
    key1: u64,
    key2: u64,
}

impl SimSipHasher64 {
    pub fn new(key1: u64, key2: u64) -> Self {
        SimSipHasher64 {
            key1,
            key2,
        }
    }
}

impl SimHasher for SimSipHasher64 {
    type T = u64;

    fn hash<U>(&self, item: &U) -> Self::T
    where
        Self: Sized,
        U: Hash,
    {
        let mut sip = SipHasher::new_with_keys(self.key1, self.key2);
        item.hash(&mut sip);
        sip.finish()
    }
}

pub struct ShaHasher64 {}

impl ShaHasher64 {
    pub fn new() -> Self {
        ShaHasher64 {}
    }
}

impl SimHasher for ShaHasher64 {
    type T = u64;

    fn hash<U>(&self, item: &U) -> Self::T
    where
        Self: Sized,
        U: Hash,
    {
        let mut hasher = Sha1Hasher::new();
        item.hash(&mut hasher);
        hasher.finish()
    }
}

pub struct SimSipHasher128 {
    key1: u64,
    key2: u64,
}

impl SimSipHasher128 {
    pub fn new(key1: u64, key2: u64) -> Self {
        SimSipHasher128 {
            key1,
            key2,
        }
    }
}

impl SimHasher for SimSipHasher128 {
    type T = u128;

    fn hash<U>(&self, item: &U) -> Self::T
    where
        Self: Sized,
        U: Hash,
    {
        let mut sip = SipHasher128::new_with_keys(self.key1, self.key2);
        item.hash(&mut sip);
        sip.finish128().as_u128()
    }
}

pub struct Xxh3Hasher64;

impl Xxh3Hasher64 {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl SimHasher for Xxh3Hasher64 {
    type T = u64;

    #[inline]
    fn hash<U>(&self, item: &U) -> Self::T
    where
        U: Hash,
    {
        let mut h = Xxh3::new();
        item.hash(&mut h);
        h.finish()
    }
}

pub struct Xxh3Hasher128;           // no fields

impl Xxh3Hasher128 {
    pub fn new() -> Self { Self }
}

impl SimHasher for Xxh3Hasher128 {
    type T = u128;

    #[inline]
    fn hash<U>(&self, item: &U) -> Self::T
    where
        U: Hash,
    {
        let mut h = Xxh3::new();    // default seed = 0
        item.hash(&mut h);
        h.digest128()               // returns u128 directly (crate ≥ 1.2)
    }
}