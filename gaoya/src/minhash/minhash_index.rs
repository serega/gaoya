use crate::minhash::{compute_minhash_distance, compute_minhash_similarity};
use fxhash::FxBuildHasher;
use fxhash::FxHashMap;
use fxhash::FxHashSet;
use rayon::prelude::*;
use std::any::type_name;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::{fmt, slice};
use std::ops::Range;
use itertools::Itertools;
use sha1::digest::generic_array::typenum::Cmp;
use crate::clustering::QueryIndex;

/*
MinHashIndex stores all minhashes as Vec<T> in a hashmap, and uses unsafe pointer arithmetic
to access the band portion of the minhash directly in the vector.

Having full simhashes is useful for computing a centroid of a cluster. The unsafe data structure
gives free access to whole minhashes, and bands without sacrificing neither performance nor
memory utilization.
 */

struct BandKey<T: Hash + Eq> {
    v: *const T,
    len: usize,
}

unsafe impl<T: Hash + Eq> Send for BandKey<T> {}
unsafe impl<T: Hash + Eq> Sync for BandKey<T> {}

impl<T: Hash + Eq> Eq for BandKey<T> {}

impl<T: Hash + Eq> PartialEq for BandKey<T> {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            for i in 0..self.len {
                if *self.v.add(i) != *other.v.add(i) {
                    return false;
                }
            }
            return true;
        };
    }
}

impl<T: Hash + Eq> Hash for BandKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H)
    where
        T: Hash,
    {
        unsafe {
            for i in 0..self.len {
                (*self.v.add(i)).hash(state)
            }
        }
    }
}

struct MinHashBand<T, Id>
where
    T: Hash + Eq,
    Id: Hash + Eq + Clone,
{
    hash_table: FxHashMap<BandKey<T>, FxHashSet<Id>>,
    band_start: isize,
    band_end: isize,
    len: usize,
}

impl<T, Id> MinHashBand<T, Id>
where
    T: Hash + Eq,
    Id: Hash + Eq + Clone,
{
    pub fn new(band_start: isize, band_end: isize) -> Self {
        let mut hash_table = FxHashMap::default();
        hash_table.reserve(1000);
        MinHashBand {
            hash_table: hash_table,
            band_start: band_start,
            band_end: band_end,
            len: (band_end - band_start) as usize,
        }
    }

    pub fn new_with_capacity(band_start: isize, band_end: isize, capacity: usize) -> Self {
        let mut hash_table = FxHashMap::default();
        hash_table.reserve(capacity);
        MinHashBand {
            hash_table: hash_table,
            band_start: band_start,
            band_end: band_end,
            len: (band_end - band_start) as usize,
        }
    }


    fn insert(&mut self, id: Id, signature: &Vec<T>) {
        let band_data = unsafe { signature.as_ptr().offset(self.band_start) };
        let band_key = BandKey {
            v: band_data,
            len: self.len,
        };
        self.hash_table
            .entry(band_key)
            .or_insert(FxHashSet::default())
            .insert(id.clone());
        ()
    }

    fn query<'a, S: BuildHasher>(&'a self, signature: &Vec<T>, match_ids: &mut HashSet<&'a Id, S>) {
        let band_data = unsafe { signature.as_ptr().offset(self.band_start) };
        let band_key = BandKey {
            v: band_data,
            len: self.len,
        };
        match self.hash_table.get(&band_key) {
            Some(ids) => match_ids.extend(ids.iter()),
            None => (),
        }
    }

    fn query_to_owned<S: BuildHasher>(&self, signature: &Vec<T>, match_ids: &mut HashSet<Id, S>) {
        let band_data = unsafe { signature.as_ptr().offset(self.band_start) };
        let band_key = BandKey {
            v: band_data,
            len: self.len,
        };
        match self.hash_table.get(&band_key) {
            Some(ids) => {
                match_ids.extend(ids.iter().cloned());
            }
            None => (),
        }
    }

    fn best_minhash_index<'a>(
        &'a self,
        signatures: &Vec<&[T]>,
        all_ids: &mut HashSet<&'a Id>,
    ) -> isize {
        let mut max_count: usize = 0;
        let mut best_index: isize = -1;
        for minhash in signatures.iter().enumerate() {
            let band_key = BandKey {
                v: minhash.1.as_ptr(),
                len: self.len,
            };
            match self.hash_table.get(&band_key) {
                Some(ids) => {
                    let new_count = ids.iter().map(|id| !all_ids.contains(&id) as usize).count();
                    if new_count > max_count {
                        max_count = new_count;
                        best_index = minhash.0 as isize;
                    }
                }
                None => (),
            }
        }
        let band_key = BandKey {
            v: signatures[best_index as usize].as_ptr(),
            len: self.len,
        };
        match self.hash_table.get(&band_key) {
            Some(ids) => {
                for id in ids {
                    all_ids.insert(id);
                }
            }
            None => (),
        }
        best_index
    }

    fn remove(&mut self, id: &Id, signature: &Vec<T>) -> bool {
        let band_data = unsafe { signature.as_ptr().offset(self.band_start) };
        let band_key = BandKey {
            v: band_data,
            len: (self.band_end - self.band_start) as usize,
        };

        match self.hash_table.get_mut(&band_key) {
            Some(ids) => {
                ids.remove(id);
                if ids.is_empty() {
                    self.hash_table.remove(&band_key);
                    return true;
                }
                false
            }
            None => false,
        }
    }

    fn clear(&mut self) {
        self.hash_table.clear();
    }

    fn has_ids(&self, signature: &Vec<T>) -> bool {
        let band_data = unsafe { signature.as_ptr().offset(self.band_start) };
        let band_key = BandKey {
            v: band_data,
            len: (self.band_end - self.band_start) as usize,
        };
        match self.hash_table.get(&band_key) {
            Some(ids) => ids.len() > 0,
            None => false,
        }
    }

    pub fn shrink_to_fit(&mut self) {
        for item in self.hash_table.iter_mut() {
            item.1.shrink_to_fit();
        }
        self.hash_table.shrink_to_fit();
    }
}

pub struct MinHashIndex<T, Id>
where
    T: Hash + Eq,
    Id: Hash + Eq + Clone,
{
    bands: Vec<MinHashBand<T, Id>>,
    removed_ids: HashSet<Id>,
    id_signatures: FxHashMap<Id, Vec<T>>,
    threshold: f64,
    r: usize,
    b: usize,
    size: usize,
}

static REMOVED_KEYS_COUNT_CLEAN_TRIGGER: usize = 1000;

impl<T, Id> fmt::Display for MinHashIndex<T, Id>
where
    T: Hash + Eq,
    Id: Hash + Eq + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinHashIndex<{}> {{ threshold = {}, num_perms = {}, bands = {}, rows_per_band = {}, size = {} }}",
               type_name::<T>(),
               self.threshold, self.b * self.r, self.b, self.r, self.size)
    }
}

impl<T, Id> MinHashIndex<T, Id>
where
    T: Hash + Eq,
    Id: Hash + Eq + Clone,
{
    pub fn new_with_weights(threshold: f64, num_perm: usize, fpw: f64, fnw: f64) -> Self {
        let (b, r) = optimal_param(threshold, num_perm, fpw, fnw);
        println!("{} {}", b, r);
        let mut bands = Vec::new();
        for i in 0..b {
            let (start, end) = (i * r, (i + 1) * r);
            bands.push(MinHashBand::<T, Id>::new(start as isize, end as isize));
        }
        let mut hash_table = FxHashMap::default();
        hash_table.reserve(1000);
        MinHashIndex {
            bands: bands,
            removed_ids: HashSet::new(),
            threshold: threshold,
            id_signatures: hash_table,
            b: b,
            r: r,
            size: 0,
        }
    }

    pub fn new(threshold: f64, num_hashes: usize) -> Self {
        MinHashIndex::new_with_weights(threshold, num_hashes, 0.5, 0.5)
    }

    pub fn new_with_params(num_bands: usize, band_width: usize, threshold: f64) -> Self {
        let mut bands = Vec::new();
        for i in 0..num_bands {
            let (start, end) = (i * band_width, (i + 1) * band_width);
            bands.push(MinHashBand::<T, Id>::new(start as isize, end as isize));
        }
        let mut hash_table = FxHashMap::default();
        hash_table.reserve(1000);
        MinHashIndex {
            bands: bands,
            removed_ids: HashSet::new(),
            threshold: threshold,
            id_signatures: hash_table,
            b: num_bands,
            r: band_width,
            size: 0,
        }
    }

    pub fn new_with_params_and_capacity(num_bands: usize, band_width: usize, threshold: f64, capacity: usize) -> Self {
        let mut bands = Vec::new();
        for i in 0..num_bands {
            let (start, end) = (i * band_width, (i + 1) * band_width);
            bands.push(MinHashBand::<T, Id>::new_with_capacity(start as isize, end as isize, capacity));
        }
        let mut hash_table = FxHashMap::default();
        hash_table.reserve(capacity);
        MinHashIndex {
            bands: bands,
            removed_ids: HashSet::new(),
            threshold: threshold,
            id_signatures: hash_table,
            b: num_bands,
            r: band_width,
            size: 0,
        }
    }

    pub fn get_keys(&self) -> Vec<Id> {
        self.bands[0].hash_table.values()
            .into_iter().flat_map(|s| s.iter())
            .map(|id| id.clone())
            .collect()
    }

    pub fn get_keys_refs(&self) -> Vec<&Id> {
        self.bands[0].hash_table.values()
            .into_iter().flat_map(|s| s.iter())
            .collect()
    }



    pub fn insert(&mut self, id: Id, signature: Vec<T>) {
        for band in &mut self.bands {
            band.insert(id.clone(), &signature);
        }
        self.id_signatures.insert(id, signature);
        self.size += 1;
    }

    pub fn par_bulk_insert(&mut self, ids: Vec<Id>, signatures: Vec<Vec<T>>)
    where
        Id: Hash + Eq + Clone + Send + Sync,
        T: Send + Sync,
    {
        unsafe {
            self.bands.par_iter_mut().for_each(|band| {
                for item in signatures.iter().zip(ids.iter()) {
                    let hashes = item.0;
                    let id = item.1.clone();
                    band.insert(id, hashes);
                }
            });
        }
        for id_hash in ids.into_iter().zip(signatures.into_iter()) {
            match self.id_signatures.insert(id_hash.0, id_hash.1) {
                None => self.size += 1,
                Some(_) => ()
            }
        }
    }

    pub fn par_bulk_insert_pairs(&mut self, id_signature_pairs: Vec<(Id, Vec<T>)>)
    where
        Id: Hash + Eq + Clone + Send + Sync,
        T: Send + Sync,
    {
        unsafe {
            self.bands.par_iter_mut().for_each(|band| {
                for item in id_signature_pairs.iter() {
                    let i: &(Id, Vec<T>) = item;
                    let (a, b) = i;
                    let k: Id = a.clone();
                    band.insert(k, &b);
                }
            });
        }
        for id_hash in id_signature_pairs {
            self.id_signatures.insert(id_hash.0, id_hash.1);
            self.size += 1;
        }
        self.id_signatures.shrink_to_fit();
    }

    pub fn shrink_to_fit(&mut self)
    where Id: Send + Sync,
          T: Send + Sync
    {
        self.bands.par_iter_mut()
            .for_each(|band| band.shrink_to_fit());
        self.id_signatures.shrink_to_fit();
    }

    pub fn clear(&mut self) {
        self.bands.iter_mut().for_each(|band| band.clear());
        self.id_signatures.clear();
    }

    pub fn query_one(&self, query_signature: &Vec<T>) -> Option<&Id> {
        let mut match_ids = HashSet::with_capacity_and_hasher(10, FxBuildHasher::default());
        for band in &self.bands {
            band.query(query_signature, &mut match_ids);
        }

        if self.removed_ids.len() > 0 {
            match_ids.retain(|item| !self.removed_ids.contains(item));
        }

        let best_match = match_ids.into_iter()
            .map(|id| {
                let signature = &self.id_signatures[id];
                (id, compute_minhash_similarity(signature, query_signature))
            })
            .filter(|pair| pair.1 > self.threshold)
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        match best_match {
            Some(pair) => Some(pair.0),
            None => None
        }
    }

    pub fn query(&self, query_signature: &Vec<T>) -> HashSet<&Id, FxBuildHasher>
    where
        Id: Hash + Eq + Clone,
    {
        let mut match_ids = HashSet::with_capacity_and_hasher(10, FxBuildHasher::default());
        for band in &self.bands {
            band.query(query_signature, &mut match_ids);
        }

        if self.removed_ids.len() > 0 {
            match_ids.retain(|item| !self.removed_ids.contains(item));
        }

        match_ids.retain(|id| {
            let signature = &self.id_signatures[id];
            compute_minhash_similarity(signature, query_signature) > self.threshold
        });

        match_ids
    }

    /*
    pub fn query_mut(&mut self, query_signature: &Vec<T>) -> HashSet<&mut Id, FxBuildHasher>
        where
            Id: Hash + Eq + Clone,
    {
        let mut match_ids = HashSet::with_capacity_and_hasher(10, FxBuildHasher::default());
        for band in &self.bands {
            band.query_mut(query_signature, &mut match_ids);
        }

        if self.removed_ids.len() > 0 {
            match_ids.retain(|item| !self.removed_ids.contains(item));
        }

        match_ids.retain(|id| {
            let signature = &self.id_signatures[id];
            compute_minhash_distance(signature, query_signature) < self.threshold
        });

        match_ids
    }

     */


    pub fn query_owned(&self, query_signature: &Vec<T>) -> HashSet<Id, FxBuildHasher>
    where
        Id: Hash + Eq + Clone,
    {
        let mut match_ids = HashSet::with_capacity_and_hasher(10, FxBuildHasher::default());
        for band in &self.bands {
            band.query_to_owned(query_signature, &mut match_ids);
        }
        if self.removed_ids.len() > 0 {
            match_ids.retain(|item| !self.removed_ids.contains(item));
        }
        match_ids.retain(|id| {
            let signature = &self.id_signatures[id];
            compute_minhash_distance(signature, query_signature) < self.threshold
        });
        match_ids
    }

    pub fn query_top_k(&self, query_signature: &Vec<T>, k: usize) -> Vec<(Id, f64)> {
        let mut match_ids = HashSet::with_capacity_and_hasher(10, FxBuildHasher::default());
        for band in &self.bands {
            band.query_to_owned(query_signature, &mut match_ids);
        }
        if self.removed_ids.len() > 0 {
            match_ids.retain(|item| !self.removed_ids.contains(item));
        }
        let mut ids_distances: Vec<(Id, f64)> = match_ids
            .into_iter()
            .map(|id| {
                let signature = &self.id_signatures[&id];
                let distance = compute_minhash_distance(query_signature, signature);
                (id, distance)
            })
            .collect();
        ids_distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ids_distances[0..std::cmp::min(ids_distances.len(), k)].to_vec()
    }

    /// Removes a id from the index.
    pub fn remove(&mut self, id: &Id) {
        match self.id_signatures.get(id) {
            Some(hashes) => {
                let fully_removed = self
                    .bands
                    .iter_mut()
                    .map(|band| band.remove(id, hashes) as usize)
                    .sum::<usize>()
                    == self.b;
                if fully_removed {
                    self.id_signatures.remove(id);
                    ()
                } else {
                    self.removed_ids.insert(id.clone());
                    ()
                }
                self.size -= 1;
            }
            None => (),
        }
        if self.removed_ids.len() > REMOVED_KEYS_COUNT_CLEAN_TRIGGER {
            self.clean_removed();
        }
    }

    fn clean_removed(&mut self) {
        let fully_removed_ids: Vec<Id> = self
            .removed_ids
            .iter()
            .filter(|id| {
                let signature = self.id_signatures.get(id).unwrap();
                self.bands
                    .iter()
                    .filter(|band| band.has_ids(signature))
                    .count()
                    == 0
            })
            .map(|id| id.clone())
            .collect();

        for id in fully_removed_ids {
            self.removed_ids.remove(&id);
            self.id_signatures.remove(&id);
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn num_perms(&self) -> usize {
        self.b * self.r
    }

    /// This method filters candidates by measuring similarity between query_minhash and
    /// each candidate using full minhash similarity measure.
    fn filter_by_minhash_similarity<'a>(
        &self,
        query_signature: &Vec<T>,
        candidates: HashSet<&'a Id>,
    ) -> HashSet<&'a Id> {
        let mut result = HashSet::new();
        for candidate in candidates {
            let candidate_signature = &self.id_signatures[candidate];
            let similarity = compute_minhash_distance(query_signature, candidate_signature);
            if similarity >= self.threshold {
                result.insert(candidate);
            }
        }
        result
    }

    pub fn calculate_centroid<S: BuildHasher>(
        &self,
        ids: &HashSet<&Id, S>,
        starting_centroid: &Vec<T>,
    ) -> Vec<T>
    where
        T: Clone,
    {
        self.calculate_centroid_by_band_majority(ids)
    }

    pub fn calculate_centroid_from_starting(
        &self,
        ids: &HashSet<&Id>,
        starting_centroid: &Vec<T>,
    ) -> Vec<T>
    where
        T: Clone,
    {
        let mut bands: Vec<HashSet<&[T]>> = Vec::new();
        for i in 0..self.b {
            bands.push(HashSet::new());
        }
        for id in ids {
            let mut signature = self.id_signatures.get(&id).unwrap();
            for i in 0..self.b {
                let band: &[T] = &signature[self.band_range(i)];
                bands[i].insert(band);
            }
        }

        let mut all_ids: HashSet<&Id> = ids.iter().map(|id| id.clone()).collect();
        let mut centroid = Vec::new();
        for i in 0..self.b {
            let band = &self.bands[i];
            let band_signatures = bands[i].iter().map(|k| *k).collect();
            let index = band.best_minhash_index(&band_signatures, &mut all_ids);
            if index >= 0 {
                centroid.extend_from_slice(&band_signatures[index as usize]);
            } else {
                let sl: &[T] = &starting_centroid[i * self.r..(i + 1) * self.r];
                centroid.extend_from_slice(sl);
            }
        }
        assert_eq!(starting_centroid.len(), centroid.len());
        centroid
    }

    pub fn calculate_centroid_(&self, ids: &HashSet<&Id>) -> Vec<T>
    where
        T: Clone
    {
        let mut bands: Vec<HashSet<&[T]>> = Vec::new();
        for i in 0..self.b {
            bands.push(HashSet::new());
        }
        for id in ids {
            let mut signature = self.id_signatures.get(&id).unwrap();
            for i in 0..self.b {
                let band: &[T] = &signature[self.band_range(i)];
                bands[i].insert(band);
            }
        }
        let mut all_ids = HashSet::new();
        let mut centroid = Vec::new();
        for i in 0..self.b {
            let band = &self.bands[i];
            let band_signatures = bands[i].iter().map(|k| *k).collect();
            let index = band.best_minhash_index(&band_signatures, &mut all_ids);
            centroid.extend_from_slice(&band_signatures[index as usize]);
        }

        centroid
    }

    fn band_range(&self, band_index: usize) -> Range<usize> {
        band_index * self.r..(band_index + 1) * self.r
    }

    pub fn calculate_centroid_by_band_majority_v(
        &self,
        ids: &Vec<Id>,
    ) -> Vec<T>
        where
            T: Clone,
    {
        let mut band_counters: Vec<HashMap<&[T], usize>> = Vec::new();
        for i in 0..self.b {
            band_counters.push(HashMap::new());
        }

        for id in ids {
            let signature = self.id_signatures.get(&id).unwrap();
            for i in 0..self.b {
                let band: &[T] = &signature[self.band_range(i)];
                let count = band_counters[i].entry(band).or_insert(1);
                *count += 1;
            }
        }

        let mut centroid = Vec::new();
        for counter in band_counters {
            let mut band_counts = counter.iter().collect::<Vec<(_, &usize)>>();
            band_counts.sort_unstable_by(|a, b| b.1.cmp(a.1));
            centroid.extend_from_slice(band_counts[0].0.clone());
        }
        centroid
    }



    pub fn calculate_centroid_by_band_majority<S: BuildHasher>(
        &self,
        ids: &HashSet<&Id, S>,
    ) -> Vec<T>
    where
        T: Clone,
    {
        let mut band_counters: Vec<HashMap<&[T], usize>> = Vec::new();
        for i in 0..self.b {
            band_counters.push(HashMap::new());
        }

        for id in ids {
            let signature = self.id_signatures.get(&id).unwrap();
            for i in 0..self.b {
                let band: &[T] = &signature[self.band_range(i)];
                let count = band_counters[i].entry(band).or_insert(1);
                *count += 1;
            }
        }

        let mut centroid = Vec::new();
        for counter in band_counters {
            let mut band_counts = counter.iter().collect::<Vec<(_, &usize)>>();
            band_counts.sort_unstable_by(|a, b| b.1.cmp(a.1));
            centroid.extend_from_slice(band_counts[0].0.clone());
        }
        centroid
    }

    pub fn calculate_centroid_by_hash_majority(&self, ids: &HashSet<&Id>) -> Vec<T>
    where
        Id: Hash + Eq + Clone,
        T: Clone + Copy,
    {
        let mut counters: Vec<HashMap<&T, usize>> = Vec::new();

        for i in 0..self.num_perms() {
            counters.push(HashMap::new());
        }

        for id in ids {
            let signature = self.id_signatures.get(&id).unwrap();
            for hash in signature.iter().enumerate() {
                let count = counters[hash.0].entry(hash.1).or_insert(1);
                *count += 1;
            }
        }

        let mut centroid = Vec::new();
        for counter in counters {
            let mut l = counter.iter().collect::<Vec<(&&T, &usize)>>();
            l.sort_unstable_by(|a, b| b.1.cmp(a.1));
            centroid.push(*l[0].0.clone());
        }
        centroid
    }
}


impl<T, Id> QueryIndex for MinHashIndex<T, Id>
    where
        T: Hash + Eq,
        Id: Hash + Eq + Clone {
    type Id = Id;

    fn query(&self, id: &Self::Id) -> HashSet<&Self::Id, FxBuildHasher> {
        match self.id_signatures.get(id) {
            Some(signature) => {
                self.query(&signature)
            }
            None => HashSet::default()
        }
    }

    /*
    fn query_mut(&self, id: &Self::Id) -> HashSet<&mut Self::Id, FxBuildHasher> {
        match self.id_signatures.get(id) {
            Some(signature) => {
                self.query(&signature)
            }
            None => HashSet::default()
        }
    }

     */
}

pub fn calculate_minhash_index_params(jaccard_distance_threshold: f64,
                                      num_perm: usize,
                                      false_positive_weight: f64,
                                      false_negative_weight: f64) -> (usize, usize) {
    optimal_param(jaccard_distance_threshold, num_perm, false_positive_weight, false_negative_weight)
}

// Calculate optimal param
// https://github.com/ekzhu/datasketch/blob/master/datasketch/lsh.py
// TODO: Understand what this code is doing
fn false_positive_proba(threshold: f64, b: usize, r: usize) -> f64 {
    let r = r as f64;
    let b = b as f64;
    return integrate(
        Box::new(|s: f64| 1.0 - (1.0 - s.powf(r)).powf(b)),
        0.0,
        threshold,
    );
}

fn false_negative_proba(threshold: f64, b: usize, r: usize) -> f64 {
    let r = r as f64;
    let b = b as f64;
    return integrate(
        Box::new(|s: f64| 1.0 - (1.0 - (1.0 - s.powf(r)).powf(b))),
        threshold,
        1.0,
    );
}

fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let p = 0.001;
    let mut area = 0.0;
    let mut x = a;
    while x < b {
        area += f(x + 0.5 * p) * p;
        x += p;
    }
    return area;
}

fn optimal_param(
    threshold: f64,
    num_perm: usize,
    false_positive_weight: f64,
    false_negative_weight: f64,
) -> (usize, usize) {
    let mut min_error = 99999999999.0;
    let mut opt = (0, 0);
    for b in 1..num_perm + 1 {
        let max_r = num_perm / b;
        for r in 1..max_r + 1 {
            let fp = false_positive_proba(threshold, b, r);
            let _fn = false_negative_proba(threshold, b, r);
            let error = fp * false_positive_weight + _fn * false_negative_weight;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}



#[cfg(test)]
mod tests {
    use super::optimal_param;
    use crate::minhash::min_hash64::MinHash64V1;
    use crate::minhash::{MinHash, MinHashIndex};
    use rand::distributions::{Distribution, Uniform};
    use rand::prelude::ThreadRng;
    use rand::{thread_rng, Rng};
    use std::borrow::Borrow;

    static S1: &'static str = "local sensitive hashing is cool";
    static S2: &'static str = "local sensitive hashing is great";
    static S3: &'static str = "local sensitive hashing is awesome";
    static S4: &'static str = "we all scream for ice cream";
    static S5: &'static str = "we all scream for ice cream sandwich";
    static S6: &'static str = "i like ice cream sandwich";

    #[test]
    pub fn test_lsh_index() {
        let min_hash = MinHash64V1::new(200);
        let mut lsh_index = MinHashIndex::new(0.5, 200);
        lsh_index.insert(1, min_hash.create_signature(S1.split_whitespace()));
        lsh_index.insert(2, min_hash.create_signature(S2.split_whitespace()));
        lsh_index.insert(3, min_hash.create_signature(S3.split_whitespace()));
        lsh_index.insert(4, min_hash.create_signature(S4.split_whitespace()));
        lsh_index.insert(5, min_hash.create_signature(S5.split_whitespace()));
        lsh_index.insert(6, min_hash.create_signature(S6.split_whitespace()));

        println!("{}", lsh_index);
        assert_eq!(lsh_index.size, 6);
        let ret = lsh_index.query(&min_hash.create_signature(S2.split_whitespace()));

        let ret_str: String = ret.iter().map(|x| x.to_string()).collect();
        assert_eq!(ret.len(), 3, "{}", ret_str);
        assert!(ret.contains(&1));
        assert!(ret.contains(&2));
        assert!(ret.contains(&3));

        lsh_index.remove(&2);
        assert_eq!(lsh_index.size, 5);
        let ret = lsh_index.query(&min_hash.create_signature(S2.split_whitespace()));
        assert_eq!(ret.len(), 2);
        assert!(ret.contains(&1));
        assert!(ret.contains(&3));
    }

    #[test]
    pub fn test_remove() {
        let mut lsh_index = MinHashIndex::new(0.4, 9);
        lsh_index.insert(1, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        lsh_index.insert(2, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        lsh_index.insert(3, vec![1, 1, 1, 2, 2, 2, 3, 4, 4]);
        lsh_index.insert(4, vec![1, 1, 1, 2, 2, 2, 3, 4, 3]);

        lsh_index.insert(5, vec![2, 2, 2, 3, 3, 3, 4, 4, 4]);
        lsh_index.insert(6, vec![3, 3, 3, 4, 4, 4, 5, 5, 5]);
        lsh_index.insert(7, vec![3, 3, 3, 4, 4, 4, 5, 5, 5]);

        let res = lsh_index.query(&vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(res, vec![1, 2, 3, 4].iter().collect());

        lsh_index.remove(&1);
        assert_eq!(lsh_index.removed_ids.len(), 1);
        let res = lsh_index.query(&vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(res, vec![2, 3, 4].iter().collect());

        lsh_index.remove(&2);
        assert_eq!(lsh_index.removed_ids.len(), 2);
        let res = lsh_index.query(&vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(res, vec![3, 4].iter().collect());

        lsh_index.remove(&5);
        assert_eq!(lsh_index.removed_ids.len(), 2);
        let res = lsh_index.query(&vec![3, 3, 3, 4, 4, 4, 5, 5, 5]);
        assert_eq!(res, vec![6, 7].iter().collect());

        lsh_index.remove(&6);
        assert_eq!(lsh_index.removed_ids.len(), 3);
        let res = lsh_index.query(&vec![3, 3, 3, 4, 4, 4, 5, 5, 5]);
        assert_eq!(res, vec![7].iter().collect());

        lsh_index.remove(&7);
        assert_eq!(lsh_index.removed_ids.len(), 3);
        assert_eq!(
            lsh_index.removed_ids,
            vec![1, 2, 6].into_iter().collect()
        );
        let res = lsh_index.query(&vec![3, 3, 3, 4, 4, 4, 5, 5, 5]);
        assert_eq!(res.len(), 0);

        lsh_index.clean_removed();
        assert_eq!(lsh_index.removed_ids.len(), 2);
        assert_eq!(lsh_index.removed_ids, vec![1, 2].into_iter().collect());

        lsh_index.remove(&3);
        lsh_index.remove(&4);
        lsh_index.clean_removed();
        assert_eq!(lsh_index.removed_ids.len(), 0);
        assert_eq!(lsh_index.size(), 0);
    }

    #[test]
    pub fn test_lsh_index_batch_construction() {
        let min_hash = MinHash64V1::new(200);
        let mut lsh_index: MinHashIndex<u64, u64> = MinHashIndex::new(0.5, 200);
        let mut signatures: Vec<(u64, Vec<u64>)> = Vec::new();
        signatures.push((1, min_hash.create_signature(S1.split_whitespace())));
        signatures.push((2, min_hash.create_signature(S2.split_whitespace())));
        signatures.push((3, min_hash.create_signature(S3.split_whitespace())));
        signatures.push((4, min_hash.create_signature(S4.split_whitespace())));
        signatures.push((5, min_hash.create_signature(S5.split_whitespace())));
        signatures.push((6, min_hash.create_signature(S6.split_whitespace())));

        lsh_index.par_bulk_insert_pairs(signatures);
        assert_eq!(lsh_index.size, 6);

        let ret = lsh_index.query(&min_hash.create_signature(S2.split_whitespace()));

        let ret_str: String = ret.iter().map(|x| x.to_string()).collect();
        assert_eq!(ret.len(), 3, "{}", ret_str);
        assert!(ret.contains(&1));
        assert!(ret.contains(&2));
        assert!(ret.contains(&3));
    }

    fn random_change_values(
        v: &Vec<u64>,
        num_changes: usize,
        num_vecs: usize,
        rng: &mut ThreadRng,
    ) -> Vec<Vec<u64>> {
        let rand_range = Uniform::from(1..100000);
        let index_rand_range = Uniform::from(0..1000);
        (0..num_vecs)
            .map(|_| {
                let indices: Vec<usize> = (0..num_changes)
                    .map(|_| index_rand_range.sample(rng))
                    .collect();
                let changes: Vec<u64> = (0..num_changes).map(|_| rand_range.sample(rng)).collect();
                let mut c = v.clone();
                assert_eq!(c.len(), v.len());
                for i in indices.iter().zip(changes.iter()) {
                    c[*i.0] = i.1.clone();
                }
                c
            })
            .collect()
    }

    #[test]
    pub fn test_lsh_index_batch_construction2() {
        let min_hash = MinHash64V1::new(128);
        let mut lsh_index: MinHashIndex<u64, u64> = MinHashIndex::new(0.5, 128);

        let mut vecs = Vec::new();
        let rand_range = Uniform::from(1..100000);
        let mut rng = thread_rng();
        let v1: Vec<u64> = (0..1000).map(|_| rand_range.sample(&mut rng)).collect();
        let v2: Vec<u64> = (0..1000).map(|_| rand_range.sample(&mut rng)).collect();
        let v3: Vec<u64> = (0..1000).map(|_| rand_range.sample(&mut rng)).collect();
        assert_eq!(v1.len(), 1000);
        vecs.push(v1.clone());
        vecs.extend_from_slice(random_change_values(&v1, 100, 99, &mut rng).as_slice());
        vecs.push(v2.clone());
        vecs.extend_from_slice(random_change_values(&v2, 50, 99, &mut rng).as_slice());
        vecs.push(v3.clone());
        vecs.extend_from_slice(random_change_values(&v3, 10, 99, &mut rng).as_slice());

        let mut ids: Vec<u64> = (0..300).collect();
        let signatures = vecs
            .iter()
            .map(|v| min_hash.create_signature(v.iter()))
            .collect();

        assert_eq!(vecs.len(), ids.len());
        lsh_index.par_bulk_insert(ids, signatures);
        assert_eq!(lsh_index.size, 300);

        let ret = lsh_index.query(&min_hash.create_signature(v1.iter()));
        assert_eq!(ret.len(), 100);
        assert_eq!((0..100).filter(|i| ret.contains(&i)).count(), 100);

        let ret = lsh_index.query_top_k(&min_hash.create_signature(v1.iter()), 10);
        assert_eq!(ret.len(), 10);
        println!("{:?}", ret);

        let ret = lsh_index.query(&min_hash.create_signature(v2.iter()));
        assert_eq!(ret.len(), 100);
        assert_eq!((100..200).filter(|i| ret.contains(&i)).count(), 100);

        let ret = lsh_index.query(&min_hash.create_signature(v3.iter()));
        assert_eq!(ret.len(), 100);
        assert_eq!((200..300).filter(|i| ret.contains(&i)).count(), 100);
    }
}
