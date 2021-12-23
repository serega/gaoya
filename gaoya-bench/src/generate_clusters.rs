use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::{seq::IteratorRandom, thread_rng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct GeneratedCluster {
    pub id: usize,
    pub points: HashMap<u32, Vec<usize>>,

    // The initial starting point from which other points were generated
    pub centroid: Vec<usize>,
}

/// DifferenceMode indicates whether the changes in the same cluster are in the same or
/// different positions.
///
/// When DifferenceMode is SameIndices we choose num_changes random indices
/// once per generated cluster, and change values in the positions corresponding to the indices
/// in all cluster examples.
///
/// When DifferenceMode is DiffIndices we choose random indices for every cluster example.
/// For example, given a list of elements [1,2,3,4,5,6] and num_changes equal to 2.
/// SameIndices - choose (1,3). In all cluster examples we changes elements 2 and 4
/// DiffIndices - choose two random indices for every cluster example.

#[derive(Debug, PartialOrd, PartialEq)]
pub enum DifferenceMode {
    SameIndices,
    DiffIndices,
}

pub struct ClusterGenerator {
    pub jaccard_similarity: f64,
    pub cluster_size: usize,
    pub point_num_values: usize,
    pub num_clusters: usize,
    pub min_value: usize,
    pub max_value: usize,
    pub num_changes: usize,
    pub mode: DifferenceMode,
}

impl ClusterGenerator {

    pub fn new(jaccard_similarity: f64, cluster_size: usize,
               point_num_values: usize, num_clusters: usize,
               min_value: usize, max_value: usize,
               mode: DifferenceMode) -> Self {

        ClusterGenerator {
            jaccard_similarity: jaccard_similarity,
            cluster_size: cluster_size,
            point_num_values: point_num_values,
            num_clusters: num_clusters,
            min_value: min_value,
            max_value: max_value,
            num_changes: 0,
            mode: mode
        }
    }

    pub fn generate(&mut self) -> Vec<GeneratedCluster> {
        let point_id_seq = AtomicU32::new(0);
        self.num_changes = self.num_changes();
        println!("num changes {}", self.num_changes);
        if self.mode == DifferenceMode::SameIndices {
            (0..self.num_clusters)
                .into_par_iter()
                .map(|i| self.generate_cluster_same_changed_indexes(&point_id_seq, i))
                .collect()
        } else {
            (0..self.num_clusters)
                .into_par_iter()
                .map(|i| self.generate_cluster_random_changed_indexes(&point_id_seq, i))
                .collect()
        }
    }

    /// Computes the maximum number of elements changed in a given set
    /// that would result in an a jaccard similarity greater than required.
    fn num_changes(&self) -> usize {
        let mut k = 1;
        let n = self.point_num_values as f64;
        while (n - k as f64) / (n + k as f64) > self.jaccard_similarity {
            k = k + 1
        }
        k - 1
    }

    pub fn generate_cluster_same_changed_indexes(
        &self,
        point_id_sequence: &AtomicU32,
        cluster_id: usize,
    ) -> GeneratedCluster {
        let mut rng = thread_rng();
        let indices_distribution = Uniform::new(0, self.point_num_values);
        let values_distribution = Uniform::new(self.min_value, self.max_value);
        let sample: Vec<usize> = values_distribution
            .sample_iter(&mut rng)
            .take(self.point_num_values)
            .collect();

        let change_indices: Vec<usize> = indices_distribution
            .sample_iter(&mut rng)
            .take(self.num_changes)
            .collect();

        let mut points = HashMap::with_capacity(self.cluster_size);
        for _ in 0..self.cluster_size {
            let mut point_items = sample.clone();
            for j in 0..self.num_changes {
                let value = values_distribution.sample(&mut rng);
                point_items[change_indices[j]] = value;
            }
            points.insert(
                point_id_sequence.fetch_add(1, Ordering::Relaxed), point_items,
            );
        }

        GeneratedCluster {
            centroid: sample,
            id: cluster_id,
            points: points,
        }
    }

    pub fn generate_cluster_random_changed_indexes(
        &self,
        point_id_sequence: &AtomicU32,
        cluster_id: usize,
    ) -> GeneratedCluster {
        let mut rng = thread_rng();
        let values_distribution = Uniform::new(self.min_value, self.max_value);
        let sample: Vec<usize> = values_distribution
            .sample_iter(&mut rng)
            .take(self.point_num_values)
            .collect();
        let indices_distribution = Uniform::new(0, self.point_num_values);

        let mut points = HashMap::with_capacity(self.cluster_size);
        points.insert(
            point_id_sequence.fetch_add(1, Ordering::Relaxed),
            sample.clone(),
        );
        for _ in 0..self.cluster_size {
            let mut point_items = sample.clone();
            for _ in 0..self.num_changes {
                let change_index = indices_distribution.sample(&mut rng);
                let value = values_distribution.sample(&mut rng);
                point_items[change_index] = value;
            }
            points.insert(
                point_id_sequence.fetch_add(1, Ordering::Relaxed),
                point_items,
            );
        }

        GeneratedCluster {
            centroid: sample,
            id: cluster_id,
            points: points,
        }
    }
}
