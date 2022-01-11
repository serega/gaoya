mod hashers;
mod min_hash16;
mod min_hash32;
mod min_hash64;

mod minhash_index;
mod string_index;

pub use self::hashers::SipHasher24BuildHasher;
pub use self::hashers::Sha1Hasher;
pub use self::min_hash16::MinHash16V1;
pub use self::min_hash32::{
    MinHash32V1, MinHash32V2, SuperMinHash32V1, SuperMinHash32V2,
};
pub use self::min_hash64::MinHash64V1;
pub use self::minhash_index::MinHashIndex;
pub use self::string_index::MinHashStringIndex;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::FromIterator;
use fxhash::FxBuildHasher;
use rayon::prelude::*;


pub trait MinHash {
    /// The data type of individual hash.
    /// This should be one of u-numeric types such as u64, u32, u16, u8
    type V: Hash + Eq + Sync + Send;

    fn create_signature<T, U>(&self, iter: T) -> Vec<Self::V>
        where
            T: Iterator<Item=U>,
            U: Hash;

    fn bulk_create_signature<U>(&self, batch: &Vec<Vec<U>>) -> Vec<Vec<Self::V>>
        where
            U: Hash + Sync,
            Self: Sync + Send {
        batch
            .par_iter()
            .map(|tokens| self.create_signature(tokens.iter()))
            .collect()
    }

    fn bulk_create_signature_refs<U>(&self, batch: &Vec<&Vec<U>>) -> Vec<Vec<Self::V>>
        where
            U: Hash + Sync,
            Self: Sync + Send {
        batch
            .par_iter()
            .map(|tokens| self.create_signature(tokens.iter()))
            .collect()
    }


    fn compute_similarity<T, U>(&self, iter_1: T, iter_2: T) -> f64
        where
            T: Iterator<Item=U>,
            U: Hash {
        compute_minhash_similarity(
            &self.create_signature(iter_1),
            &self.create_signature(iter_2),
        )
    }
}

pub fn compute_jaccard_similarity<T, U>(iter_1: T, iter_2: T) -> f32
    where
        T: Iterator<Item=U>,
        U: Hash + Eq,
{
    let h1 = HashSet::<U>::from_iter(iter_1);
    let h2 = HashSet::<U>::from_iter(iter_2);
    let intersection_len = h1.intersection(&h2).count();
    intersection_len as f32 / (h1.len() + h2.len() - intersection_len) as f32
}

pub fn compute_jaccard_distance<T, U>(iter_1: T, iter_2: T) -> f32
    where
        T: Iterator<Item=U>,
        U: Hash + Eq,
{
    1.0 - compute_jaccard_similarity(iter_1, iter_2)
}

pub fn compute_minhash_similarity<T>(min_hashes_1: &[T], min_hashes_2: &[T]) -> f64
    where
        T: Eq,
{
    assert_eq!(min_hashes_1.len(), min_hashes_2.len());
    let num_hashes = min_hashes_1.len();
    let matches: u64 = min_hashes_1
        .iter()
        .zip(min_hashes_2.iter())
        .map(|(min_hash_1, min_hash_2)| (min_hash_1 == min_hash_2) as u64)
        .sum();
    (matches as f64) / (num_hashes as f64)
}

pub fn compute_minhash_distance<T>(min_hashes_1: &[T], min_hashes_2: &[T]) -> f64
    where
        T: Eq,
{
    1.0 - compute_minhash_similarity(min_hashes_1, min_hashes_2)
}

pub fn similarity_greater_than_threshold<T>(
    min_hashes_1: &[T],
    min_hashes_2: &[T],
    threshold: f64,
) -> bool
    where
        T: Eq,
{
    assert_eq!(min_hashes_1.len(), min_hashes_2.len());
    let num_hashes = min_hashes_1.len();
    let expected_matches = (num_hashes as f64 * threshold) as u32;
    let mut num_matches: u32 = 0;
    for pair in min_hashes_1.iter().zip(min_hashes_2.iter()) {
        if pair.0 == pair.1 {
            num_matches += 1;
        }
        if num_matches >= expected_matches {
            return true;
        }
    }
    false
}

fn centroid_minhash<T>(minhashes: &Vec<Vec<T>>) -> Vec<T>
    where
        T: Hash + Copy + Eq,
{
    let mut counters = Vec::new();
    let minhash_len = minhashes[0].len();
    for i in 0..minhash_len {
        counters.push(HashMap::new());
    }
    for v in minhashes {
        for i in 0..v.len() {
            let count = counters[i].entry(v[i]).or_insert(1);
            *count += 1;
        }
    }

    let mut centroid = Vec::new();
    for counter in counters {
        let mut l = counter.iter().collect::<Vec<(&T, &u32)>>();
        l.sort_unstable_by(|a, b| b.1.cmp(a.1));
        centroid.push(l[0].0.clone());
    }
    centroid
}

fn centroid_minhash_from_refs<T>(minhashes: &Vec<&Vec<T>>) -> Vec<T>
    where
        T: Hash + Copy + Eq,
{
    let mut counters = Vec::new();
    let minhash_len = minhashes[0].len();
    for i in 0..minhash_len {
        counters.push(HashMap::new());
    }
    for v in minhashes {
        for i in 0..v.len() {
            let count = counters[i].entry(v[i]).or_insert(1);
            *count += 1;
        }
    }

    let mut centroid = Vec::new();

    for counter in counters {
        let mut l = counter.iter().collect::<Vec<(&T, &u32)>>();
        l.sort_unstable_by(|a, b| b.1.cmp(a.1));
        centroid.push(l[0].0.clone());
    }

    centroid
}


/// Calculates number of bands `b` and band width `r` (number of rows) given
/// the minimum `jaccard similarity`, number of hashes `num_hashes`, and desired
/// probability `desired_proba` of two sets with similarity > `jaccard_similarity` to
/// share a bucket
/// For more info see 3.4.2 in http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf
///
/// # Examples
///
/// ```
/// use gaoya::minhash::calculate_minhash_params;
/// let (b, r) = calculate_minhash_params(0.5, 128);
/// assert_eq!(b, 42);
/// assert_eq!(r, 3);
///
/// let (b, r) = calculate_minhash_params(0.7, 196);
///  assert_eq!(b, 39);
///  assert_eq!(r, 5);
/// ```
pub fn calculate_minhash_params(jaccard_similarity: f64, num_hashes: usize)
    -> (usize, usize) {
    calculate_b_and_r(jaccard_similarity, num_hashes, 0.99)
}

pub fn calculate_minhash_params_with_proba(jaccard_similarity: f64, num_hashes: usize, desired_proba: f64)
                                -> (usize, usize) {
    calculate_b_and_r(jaccard_similarity, num_hashes, desired_proba)
}


fn calculate_b_and_r(s: f64, n: usize, p: f64) -> (usize, usize) {
    let proba = |b, r| {
        1.0 - (1.0 - s.powf(r)).powf(b)
    };
    let mut b = n;
    let mut r = 1;
    while b > 1 {
        let r1 = r + 1;
        let b1 = n / r1;
        if proba(b1 as f64, r1 as f64) > p {
            b = b1;
            r = r1;
        } else {
            break;
        }
    }
    (b, r)
}

