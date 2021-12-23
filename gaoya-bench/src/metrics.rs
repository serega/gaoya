use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};

pub struct Metrics {
    precisions: Vec<f64>,
    recalls: Vec<f64>,
    fscores: Vec<f64>
}

#[derive(Debug)]
pub struct MetricResult {
    pub precision: f64,
    pub recall: f64,
    pub fscore: f64
}

fn fscore(precision: f64, recall: f64) -> f64 {
    2.0 / (1.0 / precision + 1.0 / recall)
}

fn mean(list: &[f64]) -> f64 {
    let sum: f64 = Iterator::sum(list.iter());
    f64::from(sum) / (list.len() as f64)
}

impl Metrics {

    pub fn new() -> Self {
        Metrics { precisions: Vec::new(), recalls: Vec::new(), fscores: Vec::new() }
    }

    pub fn update_metrics<T: Eq + Hash, S: BuildHasher>(&mut self, found: &HashSet<T, S>, reference: &HashSet<T>) {
        if found.len() == 0 && reference.len() == 0 {
            return;
        }
        let intersection: f64 = reference.iter().map(|i| (found.contains(i) as i32) as f64).sum();
        //let intersect = intersection as f64;
        let precision = if found.len() == 0 { 0.0 } else { intersection / found.len() as f64 };
        let recall = if reference.len() == 0 { 1.0 } else { intersection / reference.len() as f64 };

        self.precisions.push(precision);
        self.recalls.push(recall);
        self.fscores.push(fscore(precision, recall));
    }


    pub fn get_result(&self) -> MetricResult {
        MetricResult {
            precision: mean(&self.precisions),
            recall: mean(&self.recalls),
            fscore: mean(&self.fscores)
        }
    }

}