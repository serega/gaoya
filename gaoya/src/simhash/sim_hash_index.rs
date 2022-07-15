use crate::simhash::permutation::Permutation;
use crate::simhash::sim_hash::SimHash;
use crate::simhash::SimHashBits;
use core::mem;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::BitOrAssign;
use ahash::{AHashMap, AHashSet};

struct SimHashTable<S, Id>
where
    Id: Hash + Eq + Clone,
    S: SimHashBits,
{
    permutation: Permutation<S>,
    table: AHashMap<S, Vec<Id>>,
}

impl<S, Id> SimHashTable<S, Id>
where
    Id: Hash + Eq + Clone,
    S: SimHashBits,
{
    fn new(permutation: Permutation<S>) -> Self {
        SimHashTable {
            permutation,
            table: AHashMap::default(),
        }
    }

    fn insert(&mut self, id: Id, simhash: &S) {
        let key = *simhash & self.permutation.simple_mask;
        self.table
            .entry(key)
            .or_insert(Vec::new())
            .push(id);
    }

    fn query<'a, B: BuildHasher>(&'a self,
                                 query_signature: &S,
                                 id_signatures: &AHashMap<Id, S>,
                                 match_ids: &mut HashSet<&'a Id, B>,
                                 max_distance: usize)
    {
        let key = *query_signature & self.permutation.simple_mask;
        match self.table.get(&key) {
            Some(candidates) => {
                let matches = candidates
                    .iter()
                    .filter(|id| {
                        let signature = id_signatures.get(id).unwrap();
                        query_signature.hamming_distance(signature) < max_distance
                    });
                match_ids.extend(matches);
            }
            None => (),
        }
    }

    fn query_owned<'a, B: BuildHasher>(&'a self,
                                       query_signature: &S,
                                       id_signatures: &AHashMap<Id, S>,
                                       match_ids: &mut HashSet<Id, B>,
                                       max_distance: usize)
    {
        let key = *query_signature & self.permutation.simple_mask;
        match self.table.get(&key) {
            Some(candidates) => {
                let matches = candidates
                    .iter()
                    .filter(|id| {
                        let signature = id_signatures.get(id).unwrap();
                        query_signature.hamming_distance(signature) < max_distance
                    }).cloned();
                match_ids.extend(matches);
            }
            None => (),
        }
    }

    fn avg_bucket_count(&self) -> Option<usize> {
        let sum = self.table.values().map(|v| v.len()).sum::<usize>();
        match self.table.len() {
            len if len > 0 => Some(sum / len),
            _ => None,
        }
    }
}
unsafe impl<S: SimHashBits, Id: Hash + Eq + Clone> Send for SimHashTable<S, Id> {}
unsafe impl<S: SimHashBits, Id: Hash + Eq + Clone> Sync for SimHashTable<S, Id> {}



pub struct SimHashIndex<S, Id>
where
    S: SimHashBits,
    Id: Hash + Eq + Clone,
{
    num_blocks: usize,
    hamming_distance: usize,
    hash_tables: Vec<SimHashTable<S, Id>>,
    id_signatures: AHashMap<Id, S>,
    marker: PhantomData<(S, Id)>,
    size: usize,
}

impl<S, Id> SimHashIndex<S, Id>
where
    S: SimHashBits,
    Id: Hash + Eq + Clone,
{

    pub fn new(num_blocks: usize, hamming_distance: usize) -> Self {
        let permutations = Permutation::<S>::create(num_blocks, hamming_distance);
        let max_width: usize = permutations.iter().map(|p| p.width).max().unwrap();
        SimHashIndex {
            num_blocks,
            hamming_distance,
            hash_tables: (permutations
                .into_iter()
                .map(|permutation| SimHashTable::new(permutation))
                .collect()),
            id_signatures: AHashMap::default(),
            marker: PhantomData,
            size: 0,
        }
    }

    pub fn insert(&mut self, id: Id, signature: S) {
        for table in &mut self.hash_tables {
            table.insert(id.clone(), &signature);
        }
        self.id_signatures.insert(id, signature);
    }

    pub fn par_bulk_insert(&mut self, ids: Vec<Id>, signatures: Vec<S>)
    where
        S: Send + Sync,
        Id: Send + Sync,
    {
        self.hash_tables.par_iter_mut().for_each(|table| {
            for item in ids.iter().zip(signatures.iter()) {
                table.insert(item.0.clone(), item.1);
            }
        });

        for id_hash in ids.into_iter().zip(signatures.into_iter()) {
            self.id_signatures.insert(id_hash.0, id_hash.1);
        }
        self.id_signatures.shrink_to_fit();
    }

    pub fn par_bulk_insert_pairs(&mut self, id_signature_pairs: Vec<(Id, S)>)
    where
        S: Send + Sync,
        Id: Send + Sync,
    {
        self.hash_tables.par_iter_mut().for_each(|table| {
            for item in id_signature_pairs.iter() {
                let i: &(Id, S) = item;
                let (a, b) = i;
                let k: Id = a.clone();
                table.insert(k, b);
            }
        });

        for id_hash in id_signature_pairs {
            self.id_signatures.insert(id_hash.0, id_hash.1);
        }
        self.id_signatures.shrink_to_fit();
    }

    pub fn query(&self, query_signature: &S) -> AHashSet<&Id> {
        let mut match_ids = AHashSet::with_capacity(10);
        for table in &self.hash_tables {
            table.query(query_signature, &self.id_signatures,  &mut match_ids, self.hamming_distance);
        }
        match_ids
    }

    pub fn query_owned(&self, query_signature: &S) -> AHashSet<Id> {
        let mut match_ids = AHashSet::with_capacity(10);
        for table in &self.hash_tables {
            table.query_owned(query_signature, &self.id_signatures,  &mut match_ids, self.hamming_distance);
        }
        match_ids
    }
    pub fn query_return_distance(&self, query_signature: &S) -> Vec<(Id, usize)> {
        let ids = self.query(query_signature);
        let mut result: Vec<_> = ids.into_iter()
            .map(|id| (id.clone(), query_signature.hamming_distance(&self.id_signatures[id])))
            .collect();
        result.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        result
    }

    pub fn par_bulk_query(&self, signatures: &Vec<S>) -> Vec<AHashSet<Id>>
    where
        S: Send + Sync,
        Id: Send + Sync,
    {
        signatures.par_iter()
            .map(|signature| self.query_owned(signature))
            .collect()
    }

    pub fn par_bulk_query_return_distance(&self, signatures: &Vec<S>) -> Vec<Vec<(Id, usize)>>
        where
            S: Send + Sync,
            Id: Send + Sync,
    {
        signatures.par_iter()
            .map(|signature| self.query_return_distance(signature))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn avg_bucket_count(&self) -> usize {
        let counts: Vec<usize> = self
            .hash_tables
            .iter()
            .filter_map(|table| table.avg_bucket_count())
            .collect();
        match counts.len() {
            len if len > 0 => {
                let sum = counts.iter().sum::<usize>();
                sum / len
            }
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SimHashIndex;
    use crate::simhash::sim_hash::SimHash;
    use crate::simhash::sim_hasher::SimSipHasher64;
    use rand::distributions::{Distribution, Uniform};
    use rand::{thread_rng, Rng};

    #[test]
    pub fn test_simhash_index() {
        let mut sim_hash_index = SimHashIndex::<u64, usize>::new(8, 6);
        let doc = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        let rand_range = Uniform::from(1..1000);
        let mut rng = thread_rng();
        let mut docs = Vec::new();
        for i in 0..10 {
            let mut doc1 = doc.clone();
            doc1[0] = rand_range.sample(&mut rng);
            docs.push((i, doc1));
        }

        let sim_hash = SimHash::<SimSipHasher64, u64, 64>::new(SimSipHasher64::new(5, 6));
        for doc in docs {
            let signature = sim_hash.create_signature(doc.1.iter());
            sim_hash_index.insert(doc.0, signature);
        }

        let result = sim_hash_index.query(&sim_hash.create_signature(doc.iter()));
        println!("{:?}", result);
    }
}
