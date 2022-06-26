#![allow(unused_imports)]
mod metrics;
mod generate_clusters;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Bound::Included;
use std::str::Split;
use std::time::Instant;
use fxhash::FxBuildHasher;
use gaoya::clustering::clusterer_parallel::{Clusterer, ClusterPoint, ClusterPointInner};

use gaoya::minhash::{MinHasher32, MinHasher16, MinHashIndex, MinHasher64V1, MinHasher, SipHasher24BuildHasher, calculate_minhash_params, MinHasher8};
use rayon::prelude::*;

use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::generate_clusters::*;
use crate::metrics::Metrics;


fn run_clustering<M: MinHasher>(generated_clusters: &Vec<GeneratedCluster>,
                                minhash: M,
                                num_bands: usize, band_width: usize, jaccard_threshold: f64)
    where M::V: Clone, M: Sync + Send {
    println!("Creating index {}", std::any::type_name::<M>());
    let mut lsh = MinHashIndex::new(num_bands, band_width, jaccard_threshold);
    let mut ids = Vec::new();
    let mut vals = Vec::new();
    for cluster in generated_clusters {
        for pair in cluster.points.iter() {
            ids.push(pair.0.clone());
            vals.push(pair.1);
        }
    }
    let hashes = minhash.bulk_create_signature_refs(&vals);
    let ids = ids.par_iter()
        .map(|id| ClusterPoint::new(ClusterPointInner::new(id.clone()))).collect();
    lsh.par_bulk_insert(ids, hashes);
    println!("Starting clustering {}", lsh);
    let clusterer = Clusterer::<u32>::new(50, 10);
    let mut points: Vec<ClusterPoint<u32>> = lsh.get_keys();
    let now = Instant::now();
    let clusters = clusterer.cluster_par(&mut points, &lsh);
    let elapsed = now.elapsed();
    let total: usize = clusters.iter().map(|cluster| cluster.points.len()).sum();
    println!("Elapsed millis {}. Num clusters {}. Total points {} ", elapsed.as_millis(),  clusters.len(), total);

    let centroids: Vec<_> = clusters.iter()
        .map(|cluster| lsh.calculate_centroid(cluster.points.as_slice()))
        .collect();

    let mut centroid_index = MinHashIndex::new(num_bands, band_width, jaccard_threshold);
    for i in 0..clusters.len() {
        centroid_index.insert(clusters[i].cluster_id, centroids[i].clone());
    }

    let mut metrics = Metrics::new();
    for generated_cluster in generated_clusters {
        let signature = minhash.create_signature(generated_cluster.centroid.iter());
        let generated_cluster_ids: HashSet<u32> = generated_cluster.points.keys()
            .map(|id| id.to_owned())
            .collect();
        match centroid_index.query_one(&signature) {
            Some(cluster_id) => {
                let cluster = clusters.iter()
                    .find(|cl| cl.cluster_id == *cluster_id.0)
                    .unwrap();
                let cluster_ids: HashSet<u32> = cluster.points.iter()
                    .map(|p| p.id.clone())
                    .collect();
                metrics.update_metrics(&cluster_ids, &generated_cluster_ids);
            }
            None => {
                metrics.update_metrics(&HashSet::new(), &generated_cluster_ids);
            }
        }
    }
    println!("{:?}", metrics.get_result());
}


fn main() {
    let mut generator = ClusterGenerator::new(0.6, 200, 30, 500, 0, 300_000, DifferenceMode::SameIndices);
    let generated_clusters = generator.generate();
    println!("Generated {} clusters", generated_clusters.len());
    let params = (50, 5);
    println!("{:?}", params);

    run_clustering(&generated_clusters, MinHasher8::new(params.0 * params.1), params.0, params.1, 0.6);
    run_clustering(&generated_clusters, MinHasher16::new(params.0 * params.1), params.0, params.1, 0.6);
    run_clustering(&generated_clusters, MinHasher32::new(params.0 * params.1), params.0, params.1, 0.6);
}