use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Uniform};
use crate::minhash::hashers::Hashers;
use std::collections::HashMap;
use rayon::prelude::*;
use crate::minhash::compute_minhash_similarity;

pub trait MinHash64 {

    fn create_signature<T, U>(&self, iter: T) -> Vec<u64>
        where
            T: Iterator<Item=U>,
            U: Hash;

    fn bulk_create_signature<U>(&self, batch: &Vec<Vec<U>>) -> Vec<Vec<u64>>
        where
            U: Hash + Sync, Self: Sync  {
        batch.par_iter().map(|tokens| self.create_signature(tokens.iter())).collect()
    }

    fn compute_similarity<T, U>(&self, iter_1: T, iter_2: T) -> f64
        where
            T: Iterator<Item=U>,
            U: Hash,
    {
        compute_minhash_similarity(&self.create_signature(iter_1), &self.create_signature(iter_2))
    }
}


#[derive(Clone)]
pub struct MinHash64V1 {
    hashers: Hashers,
    a: Vec<u64>,
    b: Vec<u64>,
    num_hashes: usize,
}


static MERSENNE_PRIME: u64 = (1 << 61) - 1;

impl MinHash64V1 {

    /// Constructs a new `MinHash64V1` with a specified number of hash functions to use.
    /// ```
    /// use gaoya::minhash::MinHash64V1;
    ///
    /// let min_hash = MinHash64V1::new(100);
    /// ```
    pub fn new(num_hashes: usize) -> Self {
        return MinHash64V1::new_with_hasher(num_hashes, Hashers::Sip);
    }

    pub fn new_with_hasher(num_hashes: usize, hashers: Hashers) -> Self {
        let mut rng = thread_rng();
        let rand_range1 = Uniform::from(1..MERSENNE_PRIME);
        let rand_range2 = Uniform::from(0..MERSENNE_PRIME);
        MinHash64V1 {
            hashers: hashers,
            a: (0..num_hashes).map(|_| rand_range1.sample(&mut rng)).collect(),
            b: (0..num_hashes).map(|_| rand_range2.sample(&mut rng)).collect(),
            num_hashes
        }
    }

    pub fn get_hasher(&self) -> &Hashers {
        &self.hashers
    }

    /// Returns the number of hash functions being used in `MinHash`.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    fn get_min_hashes_into_vec<T, U>(&self, iter: T, ret: &mut Vec<u64>)
        where
            T: Iterator<Item = U>,
            U: Hash,
    {
        let hashes: Vec<u64> = iter.map(|item| {
            let mut hasher = self.hashers.new_hasher();
            item.hash(&mut hasher);
            hasher.finish()
        }).collect::<Vec<_>>();

        if hashes.len() > 0 {
            for index in 0..self.num_hashes {
                let m = hashes.iter().map(|hash| {
                    hash.wrapping_mul(self.a[index]).wrapping_add(self.b[index]) % MERSENNE_PRIME
                }).min().unwrap();
                ret[index] = m;
            }
        }
    }

}

impl MinHash64 for MinHash64V1 {

    fn create_signature<T, U>(&self, iter: T) -> Vec<u64>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let hashes: Vec<u64> = iter.map(|item| {
            let mut hasher = self.hashers.new_hasher();
            item.hash(&mut hasher);
            hasher.finish()
        }).collect::<Vec<_>>();

        match hashes.len() {
            len if len > 0 => {
                self.a.iter().zip(self.b.iter()).map(|ab| {
                    hashes.iter().map(|hash| {
                        hash.wrapping_mul(*ab.0).wrapping_add(*ab.1) % MERSENNE_PRIME
                    }).min().unwrap()
                }).collect()
            },
            _ => vec![0; self.num_hashes]
        }

    }

    fn bulk_create_signature<U>(&self, batch: &Vec<Vec<U>>) -> Vec<Vec<u64>>
        where
            U: Hash + Sync
    {
        batch.par_iter().map(|tokens| self.create_signature(tokens.iter())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::MinHash64V1;

    use std::f64;
    use crate::minhash::{compute_jaccard_similarity, MinHash64, centroid_minhash};
    use crate::text::tokenize_text;

    static S1: &'static str = "local sensitive hashing is cool";
    static S2: &'static str = "local sensitive hashing is great";
    static S3: &'static str = "local sensitive hashing is awesome";

    static S10: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Gail's assistant w/eXp Realty";
    static S11: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Elle's assistant w/Bright Birch Real Estate";

    #[test]
    fn test_min_hash_similarity() {
        let min_hash: MinHash64V1 = MinHash64V1::new(200);
        let similarity = min_hash.compute_similarity(tokenize_text(S10), tokenize_text(S11)) as f32;
        let actual_similarity = compute_jaccard_similarity(tokenize_text(S10), tokenize_text(S11));
        println!("actual {} estimated {} ", actual_similarity,  similarity);
        assert!(f32::abs(similarity - 0.75) < 0.1);

        let estimated_similarity = min_hash.compute_similarity(tokenize_text(S1), tokenize_text(S3)) as f32;
        let actual_similarity = compute_jaccard_similarity(tokenize_text(S1), tokenize_text(S3));
        println!("actual {} estimated {}", actual_similarity,  estimated_similarity);
        assert!(f32::abs(estimated_similarity - actual_similarity) < 0.1);
    }

    #[test]
    fn test_min_hash_centroid() {
        let min_hashes = vec![vec![1,2,3,4], vec![1,2,3,40],vec![1,20,3,40], vec![1,2,3,50]];

        let centroid = centroid_minhash(&min_hashes);
        assert_eq!(vec![1,2,3,40], centroid);
    }

}
