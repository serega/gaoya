use crate::minhash::{compute_minhash_similarity, MinHasher};
use rand::distributions::{Distribution, Uniform};

use rayon::prelude::*;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use fnv::{FnvBuildHasher, FnvHasher};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};


pub struct MinHasher16V1<B: BuildHasher> {
    build_hasher: B,
    a: Vec<u32>,
    b: Vec<u32>,
    num_hashes: usize
}

const MERSENNE_PRIME_31: u32 = (1 << 31) - 1;

impl MinHasher16V1<FnvBuildHasher> {
    pub fn new(num_hashes: usize) -> Self {
        return Self::new_with_hasher(num_hashes, FnvBuildHasher::default());
    }
}

impl<B: BuildHasher> MinHasher16V1<B> {

    pub fn new_with_hasher(num_hashes: usize, build_hasher: B) -> Self {
        Self::new_with_hasher_and_seed(num_hashes, build_hasher, 3)
    }

    pub fn new_with_hasher_and_seed(num_hashes: usize, build_hasher: B, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rand_range1 = Uniform::from(1..MERSENNE_PRIME_31 - 1);
        let rand_range2 = Uniform::from(0..MERSENNE_PRIME_31 - 1 );
        MinHasher16V1 {
            build_hasher: build_hasher,
            a: (0..num_hashes)
                .map(|_| rand_range1.sample(&mut rng))
                .collect(),
            b: (0..num_hashes)
                .map(|_| rand_range2.sample(&mut rng))
                .collect(),
            num_hashes
        }
    }



    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

impl<B: BuildHasher> MinHasher for MinHasher16V1<B> {
    type V = u16;
    fn create_signature<T, U>(&self, iter: T) -> Vec<u16>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let hashes: Vec<u32> = iter
            .map(|item| {
                let mut hasher = self.build_hasher.build_hasher();
                item.hash(&mut hasher);
                hasher.finish() as u32 % MERSENNE_PRIME_31
            })
            .collect::<Vec<_>>();

        match hashes.len() {
            len if len > 0 => self.a.iter().zip(self.b.iter())
                .map(|ab| {
                    hashes.iter()
                        .map(|hash| {
                            (hash.wrapping_mul(*ab.0).wrapping_add(*ab.1) % MERSENNE_PRIME_31)

                        })
                        .min().unwrap() as u16
                })
                .collect(),
            _ => vec![0; self.num_hashes],
        }
    }
}



#[cfg(test)]
mod tests {

    use crate::minhash::{centroid_minhash, compute_jaccard_similarity, MinHasher, MinHasher16V1};
    use crate::text::whitespace_split;
    use std::f64;

    static S1: &'static str = "local sensitive hashing is cool";
    static S2: &'static str = "local sensitive hashing is great";
    static S3: &'static str = "local sensitive hashing is awesome";

    static S10: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Gail's assistant w/eXp Realty";
    static S11: &'static str = "If you're still searching, we can visit a few open houses together in the next few weeks. It might help give clarity on what you're looking for. What do you think? - Elle's assistant w/Bright Birch Real Estate";

    #[test]
    fn test_min_hash_similarity() {
        let min_hash = MinHasher16V1::new(200);
        let similarity = min_hash.compute_similarity(whitespace_split(S10), whitespace_split(S11)) as f32;
        let actual_similarity = compute_jaccard_similarity(whitespace_split(S10), whitespace_split(S11));
        println!("actual {} estimated {} ", actual_similarity, similarity);
        assert!(f32::abs(similarity - 0.8) < 0.1);

        let estimated_similarity =
            min_hash.compute_similarity(whitespace_split(S1), whitespace_split(S3)) as f32;
        let actual_similarity = compute_jaccard_similarity(whitespace_split(S1), whitespace_split(S3));
        println!(
            "actual {} estimated {}",
            actual_similarity, estimated_similarity
        );
        assert!(f32::abs(estimated_similarity - actual_similarity) < 0.1);
    }

    #[test]
    fn test_min_hash_centroid() {
        let min_hashes = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 40],
            vec![1, 20, 3, 40],
            vec![1, 2, 3, 50],
        ];

        let centroid = centroid_minhash(&min_hashes);
        assert_eq!(vec![1, 2, 3, 40], centroid);
    }
}
