use rand::{thread_rng, Rng, SeedableRng};
use rand_pcg::Pcg32;
use std::hash::{Hash, Hasher};

use crate::minhash::compute_minhash_similarity;
use crate::minhash::hashers::Hashers;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashMap;

pub trait MinHash32 {
    fn create_signature<T, U>(&self, iter: T) -> Vec<u32>
    where
        T: Iterator<Item = U>,
        U: Hash;

    fn bulk_create_signature<U>(&self, batch: &Vec<Vec<U>>) -> Vec<Vec<u32>>
    where
        U: Hash + Sync,
        Self: Sync,
    {
        batch
            .par_iter()
            .map(|tokens| self.create_signature(tokens.iter()))
            .collect()
    }

    fn compute_similarity<T, U>(&self, iter_1: T, iter_2: T) -> f64
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        compute_minhash_similarity(
            &self.create_signature(iter_1),
            &self.create_signature(iter_2),
        )
    }
}

pub struct MinHash32V1 {
    hashers: Hashers,
    a: Vec<u32>,
    b: Vec<u32>,
    num_hashes: usize,
}

const MERSENNE_PRIME_31: u32 = (1 << 31) - 1;

impl MinHash32V1 {
    pub fn new(num_hashes: usize) -> Self {
        return MinHash32V1::new_with_hasher(num_hashes, Hashers::Sip);
    }

    pub fn new_with_hasher(num_hashes: usize, hashers: Hashers) -> Self {
        let mut rng = thread_rng();
        let rand_range1 = Uniform::from(1..MERSENNE_PRIME_31);
        let rand_range2 = Uniform::from(0..MERSENNE_PRIME_31);
        MinHash32V1 {
            hashers: hashers,
            a: (0..num_hashes)
                .map(|_| rand_range1.sample(&mut rng))
                .collect(),
            b: (0..num_hashes)
                .map(|_| rand_range2.sample(&mut rng))
                .collect(),
            num_hashes,
        }
    }

    pub fn get_hasher(&self) -> &Hashers {
        &self.hashers
    }

    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

impl MinHash32 for MinHash32V1 {
    fn create_signature<T, U>(&self, iter: T) -> Vec<u32>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let hashes: Vec<u32> = iter
            .map(|item| {
                let mut hasher = self.hashers.new_hasher();
                item.hash(&mut hasher);
                hasher.finish() as u32
            })
            .collect::<Vec<_>>();

        match hashes.len() {
            len if len > 0 => self
                .a
                .iter()
                .zip(self.b.iter())
                .map(|ab| {
                    hashes
                        .iter()
                        .map(|hash| {
                            hash.wrapping_mul(*ab.0).wrapping_add(*ab.1) % MERSENNE_PRIME_31
                        })
                        .min()
                        .unwrap()
                })
                .collect(),
            _ => vec![0; self.num_hashes],
        }
    }
}

////////////////       MinHash32V2    ///////////////////////
pub struct MinHash32V2 {
    hashers: Hashers,
    a: Vec<u64>,
    b: Vec<u64>,
    num_hashes: usize,
}

const MERSENNE_PRIME_61: u64 = (1 << 61) - 1;

impl MinHash32V2 {
    pub fn new(num_hashes: usize) -> Self {
        return MinHash32V2::new_with_hasher(num_hashes, Hashers::Sip);
    }

    pub fn new_with_hasher(num_hashes: usize, hashers: Hashers) -> Self {
        let mut rng = thread_rng();
        let rand_range1 = Uniform::from(1..u32::MAX as u64);
        let rand_range2 = Uniform::from(0..u32::MAX as u64);
        MinHash32V2 {
            hashers: hashers,
            a: (0..num_hashes)
                .map(|_| rand_range1.sample(&mut rng))
                .collect(),
            b: (0..num_hashes)
                .map(|_| rand_range2.sample(&mut rng))
                .collect(),
            num_hashes,
        }
    }

    pub fn get_hasher(&self) -> &Hashers {
        &self.hashers
    }
}

impl MinHash32 for MinHash32V2 {
    fn create_signature<T, U>(&self, iter: T) -> Vec<u32>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let hashes: Vec<u64> = iter
            .map(|item| {
                let mut hasher = self.hashers.new_hasher();
                item.hash(&mut hasher);
                hasher.finish()
            })
            .collect::<Vec<_>>();

        match hashes.len() {
            len if len > 0 => {
                self.a
                    .iter()
                    .zip(self.b.iter())
                    .map(|ab| {
                        hashes
                            .iter()
                            .map(|hash| {
                                let x = hash.wrapping_mul(*ab.0).wrapping_add(*ab.1);
                                (x % MERSENNE_PRIME_61) as u32
                                //((x & MERSENNE_PRIME_61) + (x >> 61)) as u32
                            })
                            .min()
                            .unwrap()
                    })
                    .collect()
            }
            _ => vec![0; self.num_hashes],
        }
    }
}

// Experimental
// SuperMinHash – A New Minwise Hashing Algorithm for Jaccard Similarity Estimation
// https://arxiv.org/pdf/1706.05698.pdf

pub struct SuperMinHash32V1 {
    hashers: Hashers,
    num_hashes: usize,
}

impl SuperMinHash32V1 {
    pub fn new(num_hashes: usize) -> Self {
        return SuperMinHash32V1::new_with_hasher(num_hashes, Hashers::Sip);
    }

    pub fn new_with_hasher(num_hashes: usize, hashers: Hashers) -> Self {
        SuperMinHash32V1 {
            hashers: hashers,
            num_hashes,
        }
    }

    pub fn get_hasher(&self) -> &Hashers {
        &self.hashers
    }
}

impl MinHash32 for SuperMinHash32V1 {
    fn create_signature<T, U>(&self, iter: T) -> Vec<u32>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let mut minhash = vec![99999999f32; self.num_hashes];
        for item in iter {
            let mut hasher = self.hashers.new_hasher();
            item.hash(&mut hasher);
            let h = hasher.finish();
            let mut rng = Pcg32::seed_from_u64(h);
            let mut p: Vec<u32> = (0..(self.num_hashes) as u32).collect();
            p.shuffle(&mut rng);
            let rand_range = Uniform::from(0f32..1.0f32);
            for j in (0..self.num_hashes) {
                let r = rand_range.sample(&mut rng);
                let x = minhash[j].min(r + p[j] as f32);
                minhash[j] = x;
            }
        }
        minhash.into_iter().map(|h| h as u32).collect()
    }
}

pub struct SuperMinHash32V2 {
    hashers: Hashers,
    num_hashes: usize,
}

impl SuperMinHash32V2 {
    pub fn new(num_hashes: usize) -> Self {
        return SuperMinHash32V2::new_with_hasher(num_hashes, Hashers::Sip);
    }

    pub fn new_with_hasher(num_hashes: usize, hashers: Hashers) -> Self {
        SuperMinHash32V2 {
            hashers: hashers,
            num_hashes,
        }
    }

    pub fn get_hasher(&self) -> &Hashers {
        &self.hashers
    }
}

impl MinHash32 for SuperMinHash32V2 {
    fn create_signature<T, U>(&self, iter: T) -> Vec<u32>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let mut h = vec![99999999f32; self.num_hashes];
        let m = self.num_hashes;
        let mut a = m - 1;
        let unit_range = Uniform::<f32>::new(0., 1.);
        let mut q = vec![0; m];
        let mut p: Vec<usize> = vec![0; m];
        let mut b: Vec<isize> = vec![-1; m];
        b[m - 1] = m as isize;

        for item in iter.enumerate() {
            let mut hasher = self.hashers.new_hasher();
            item.1.hash(&mut hasher);
            let mut rng = Pcg32::seed_from_u64(hasher.finish());
            let mut j: usize = 0;
            let i = item.0;
            while j <= a {
                let r: f32 = unit_range.sample(&mut rng);
                let k = Uniform::<usize>::new(j, m).sample(&mut rng);
                if q[j] != i {
                    q[j] = i;
                    p[j] = j;
                }
                if q[k] != i {
                    q[k] = i;
                    p[k] = k;
                }
                let tmp_swap = p[j];
                p[j] = p[k];
                p[k] = tmp_swap;
                let rpj = r + (j as f32);
                if rpj < h[p[j]] {
                    let j2 = min(h[p[j]] as usize, m - 1);
                    h[p[j]] = rpj;
                    if j < j2 {
                        b[j2] -= 1;
                        b[j] += 1;
                        while b[a] == 0 {
                            a -= 1;
                        }
                    }
                }
                j += 1;
            }
        }
        h.into_iter().map(|h| h as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::MinHash32;
    use super::MinHash32V1;
    use super::MinHash32V2;
    use super::SuperMinHash32V1;
    use super::SuperMinHash32V2;

    use crate::minhash::compute_jaccard_similarity;
    use crate::minhash::compute_minhash_similarity;
    use crate::text::tokenize_text;
    use std::cmp::min;
    use std::f64;

    static S1: &'static str = "local sensitive hashing is cool";
    static S2: &'static str = "local sensitive hashing is great";
    static S3: &'static str = "local sensitive hashing is awesome";

    static S10: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Gail's assistant w/eXp Realty";
    static S11: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Elle's assistant w/Bright Birch Real Estate";

    #[test]
    fn test_min_hash_v1() {
        let min_hash = MinHash32V1::new(128);
        test_min_hash(&min_hash);
    }

    #[test]
    fn test_min_hash_v2() {
        let min_hash = MinHash32V2::new(128);
        test_min_hash(&min_hash);
    }

    #[test]
    fn test_super_min_hash_v1() {
        let min_hash = SuperMinHash32V1::new(128);
        test_min_hash(&min_hash);
    }

    #[test]
    fn test_super_min_hash_v2() {
        let min_hash = SuperMinHash32V2::new(128);
        test_min_hash(&min_hash);
    }

    fn test_min_hash<M: MinHash32>(min_hash: &M) {
        let similarity = min_hash.compute_similarity(tokenize_text(S10), tokenize_text(S11)) as f32;
        let actual_similarity = compute_jaccard_similarity(tokenize_text(S10), tokenize_text(S11));
        println!("actual {} estimated {} ", actual_similarity, similarity);
        assert!(f32::abs(similarity - 0.75) < 0.15);

        let estimated_similarity =
            min_hash.compute_similarity(tokenize_text(S1), tokenize_text(S3)) as f32;
        let actual_similarity = compute_jaccard_similarity(tokenize_text(S1), tokenize_text(S3));
        println!(
            "actual {} estimated {}",
            actual_similarity, estimated_similarity
        );
        assert!(f32::abs(estimated_similarity - actual_similarity) < 0.15);
    }
}
