use crate::preprocessing::ProcessedPatientRecord;
use std::collections::HashMap;

#[derive(Default)]
struct ClassStats {
    mean: Vec<f32>,
    variance: Vec<f32>,
    prior: f32,
}

pub struct GaussianNB {
    stats: HashMap<u8, ClassStats>,
}

impl super::Model for GaussianNB {
    fn train(&mut self, data: &[ProcessedPatientRecord]) {
        if data.is_empty() {
            return;
        }

        let mut separated_by_class: HashMap<u8, Vec<&ProcessedPatientRecord>> = HashMap::new();
        for record in data {
            separated_by_class
                .entry(record.target)
                .or_default()
                .push(record);
        }

        for (class_value, class_data) in separated_by_class.iter() {
            let num_features = class_data[0].features.len();
            let mut class_stats = ClassStats::default();

            class_stats.prior = class_data.len() as f32 / data.len() as f32;

            for i in 0..num_features {
                let feature_values: Vec<f32> = class_data.iter().map(|r| r.features[i]).collect();
                let sum: f32 = feature_values.iter().sum();
                let mean = sum / feature_values.len() as f32;
                class_stats.mean.push(mean);

                let variance: f32 = feature_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>()
                    / (feature_values.len() - 1) as f32;
                class_stats.variance.push(variance + 1e-9);
            }
            self.stats.insert(*class_value, class_stats);
        }
    }

    fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        let mut best_class = 0;
        let mut max_posterior = f32::NEG_INFINITY;

        for (class_value, class_stats) in self.stats.iter() {
            let mut posterior = class_stats.prior.ln();
            for i in 0..record.features.len() {
                let likelihood = Self::calculate_likelihood(
                    record.features[i],
                    class_stats.mean[i],
                    class_stats.variance[i],
                );
                // Add a small epsilon to avoid log(0)
                let log_likelihood = (likelihood + 1e-10).ln();
                posterior += log_likelihood;
            }

            if posterior > max_posterior {
                max_posterior = posterior;
                best_class = *class_value;
            }
        }
        best_class
    }
}

impl GaussianNB {
    pub fn new() -> Self {
        GaussianNB {
            stats: HashMap::new(),
        }
    }

    fn calculate_likelihood(x: f32, mean: f32, variance: f32) -> f32 {
        let exponent = -((x - mean).powi(2)) / (2.0 * variance);
        (1.0 / (2.0 * std::f32::consts::PI * variance).sqrt()) * exponent.exp()
    }
}
