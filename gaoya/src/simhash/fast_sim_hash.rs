// Faithful Rust port of Dynatrace’s  “10× faster SimHash” bit-hack
// see original post here: https://www.dynatrace.com/engineering/blog/speeding-up-simhash-by-10x-using-a-bit-hack/
// and Java reference implementation here: https://github.com/dynatrace-oss/hash4j/blob/main/src/main/java/com/dynatrace/hash4j/similarity/FastSimHashPolicy_v1.java
// unsigned packed counters
// single Vec<u64> for the temporary lanes

use core::marker::PhantomData;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use std::hash::Hash;

use crate::simhash::sim_hasher::SimHasher;
use crate::simhash::SimHashBits;

/// number of u64 lanes needed for a signature of length `L`
const fn tmp_len<const L: usize, const B: usize>() -> usize {
    (L + (63 >> (6 - B))) >> B
}

/// repeating …001001 mask (low bit in every packed counter)
const fn bulk_mask<const B: usize>() -> u64 {
    let mut m = 1u64;
    let mut i = 0;
    while i < B {
        m |= m << (1 << (5 - i));
        i += 1;
    }
    m
}

pub struct FastSimHash<H, S, const L: usize, const BULK: usize = 3>
where
    H: SimHasher<T = u64>,
    S: SimHashBits,
{
    hasher: H,
    _p:     PhantomData<S>,
}

impl<H, S, const L: usize, const BULK: usize> FastSimHash<H, S, L, BULK>
where
    H: SimHasher<T = u64>,
    S: SimHashBits,
{
    // constants identical to the Java reference
    const PER_LANE: usize = 1 << BULK;
    const BITS_PER_COUNTER: usize = 1 << (6 - BULK);            // 8, 4, 2, …
    const BULK_MASK: u64 = bulk_mask::<BULK>();         // …001001…
    const TMP_LIMIT: u64 =
        (1u64 << 1 << (Self::BITS_PER_COUNTER - 1)) - 1;        // 255/15/3/1

    #[inline(always)]
    const fn tmp_len() -> usize { tmp_len::<L, BULK>() }

    pub fn new(hasher: H) -> Self {
        assert!(L <= S::bit_length(), "signature length too large for container");
        Self { hasher, _p: PhantomData }
    }

    pub fn create_signature<T, U>(&self, iter: T) -> S
    where
        T: IntoIterator<Item = U>,
        U: Hash,
    {
        let mut counts: [i32; L] = [0; L];
        let mut tmp: Vec<u64> = vec![0; Self::tmp_len()];

        let num_chunks = tmp.len() >> (6 - BULK);           // full 64-bit groups
        let num_tail_lanes = tmp.len() & (0x3f >> BULK);

        let mut processed_block = 0u64;
        let mut total_elements = 0u32;

        // per-feature loop
        for feat in iter {
            total_elements += 1;

            let seed = self.hasher.hash(&feat);
            let mut rng  = Xoroshiro128PlusPlus::seed_from_u64(seed);

            // full 64-bit chunks (update 8 counters when BULK=3)
            for h in 0..num_chunks {
                let rnd = rng.next_u64();
                let off = h << (6 - BULK);
                for j in 0..Self::BITS_PER_COUNTER {
                    tmp[off + j] += (rnd >> j) & Self::BULK_MASK; // *add only 1-bits*
                }
            }
            // tail lanes
            if num_tail_lanes != 0 {
                let rnd = rng.next_u64();
                let off = num_chunks << (6 - BULK);
                for j in 0..num_tail_lanes {
                    tmp[off + j] += (rnd >> j) & Self::BULK_MASK;
                }
            }

            processed_block += 1;
            if processed_block == Self::TMP_LIMIT {
                Self::flush::<L, BULK>(&mut counts, &mut tmp);
                processed_block = 0;
            }
        }
        Self::flush::<L, BULK>(&mut counts, &mut tmp);

        // threshold to signature
        let limit = (total_elements >> 1) as i32;
        let mut sig = S::zero();
        for i in 0..L {
            if counts[i] + ((i & (!total_elements as usize & 1)) as i32) > limit {
                sig |= S::one() << i;
            }
        }
        sig
    }

    // merge packed tmp counters into final signed counts
    #[inline(always)]
    fn flush<const LL: usize, const B: usize>(acc: &mut [i32; LL], tmp: &mut [u64]) {
        let per : usize = 1 << B;
        let width : usize = 1 << (6 - B);
        let mask  : u64   = (1u64 << width) - 1;

        // full bundles
        let full = LL >> B;
        for h in 0..full {
            let t = core::mem::take(&mut tmp[h]);
            let off = h << B;
            for g in 0..per {
                acc[off + g] += ((t >> (g * width)) & mask) as i32;
            }
        }
        // tail bundle
        for h in full..tmp.len() {
            let t = core::mem::take(&mut tmp[h]);
            let off = h << B;
            for g in 0..(LL - off) {
                acc[off + g] += ((t >> (g * width)) & mask) as i32;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::simhash::{BitArray, SimHashBits};
    use crate::simhash::sim_hasher::Xxh3Hasher64;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::time::Instant;
    #[test]
    fn fast_simhash_bitarray() {
        type Bits = BitArray<16>;
        const L: usize = 1024;
        const N: usize = 100_000;

        let mut rng  = StdRng::seed_from_u64(41);
        let data1: Vec<u8> = (0..N).map(|_| rng.gen_range(0..=1)).collect();
        let mut data2       = data1.clone();
        for i in (0..N).step_by(4) { data2[i] ^= 1; }

        // ground-truth cosine and angle
        let (mut dot, mut n1, mut n2) = (0f64, 0f64, 0f64);
        for i in 0..N {
            let x = data1[i] as f64;
            let y = data2[i] as f64;
            dot += x * y;
            n1  += x * x;
            n2  += y * y;
        }
        let cosine = (dot / (n1.sqrt() * n2.sqrt())).clamp(-1.0, 1.0);
        let theta  = cosine.acos(); 
        let p_bit  = theta / std::f64::consts::PI; // Charikar: P(bit differs)

        // 1-σ acceptance band for
        let mean   = p_bit * L as f64;
        let sigma  = (L as f64 * p_bit * (1.0 - p_bit)).sqrt();
        let low    = (mean - 2.0 * sigma).round() as usize;
        let high   = (mean + 2.0 * sigma).round() as usize;

        let fsh = FastSimHash::<Xxh3Hasher64, Bits, L>::new(Xxh3Hasher64::new());
        let t1 = Instant::now();
        let h1  = fsh.create_signature((0..N).map(|i| (i as u64, data1[i])));
        let h2  = fsh.create_signature((0..N).map(|i| (i as u64, data2[i])));
        let hd  = h1.hamming_distance(&h2);
        let dur = t1.elapsed();
        println!("fast SimHash: {:?}", dur);
        println!("HD = {hd}, expected ≈ {low}–{high}  (p_bit ≈ {:.3})", p_bit);
        println!("expected {:.3}", mean);
        assert!((low..=high).contains(&hd));
    }
}