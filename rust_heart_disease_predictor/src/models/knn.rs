use crate::preprocessing::ProcessedPatientRecord;
use std::collections::HashMap;

pub struct KNN {
    training_data: Vec<ProcessedPatientRecord>,
    k: usize,
}

impl super::Model for KNN {
    fn train(&mut self, training_data: &[ProcessedPatientRecord]) {
        self.training_data = training_data.to_vec();
    }

    fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        if self.training_data.is_empty() {
            return 0; // Default prediction if no training data
        }

        let mut distances = Vec::new();

        for train_record in &self.training_data {
            let distance = self.euclidean_distance(&record.features, &train_record.features);
            distances.push((distance, train_record.target));
        }

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Get the k nearest neighbors
        let k_nearest: Vec<u8> = distances
            .iter()
            .take(self.k)
            .map(|(_, target)| *target)
            .collect();

        // Return the majority vote
        self.majority_vote(&k_nearest)
    }
}

impl KNN {
    pub fn new(k: usize) -> Self {
        KNN {
            training_data: Vec::new(),
            k,
        }
    }

    pub fn train(&mut self, training_data: &[ProcessedPatientRecord]) {
        self.training_data = training_data.to_vec();
    }

    pub fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        if self.training_data.is_empty() {
            return 0; // Default prediction if no training data
        }

        let mut distances = Vec::new();

        for train_record in &self.training_data {
            let distance = self.euclidean_distance(&record.features, &train_record.features);
            distances.push((distance, train_record.target));
        }

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Get the k nearest neighbors
        let k_nearest: Vec<u8> = distances
            .iter()
            .take(self.k)
            .map(|(_, target)| *target)
            .collect();

        // Return the majority vote
        self.majority_vote(&k_nearest)
    }

    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // Handle mismatched dimensions
        }

        let sum_of_squares: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        sum_of_squares.sqrt()
    }

    fn majority_vote(&self, neighbors: &[u8]) -> u8 {
        let mut vote_counts = HashMap::new();

        for &vote in neighbors {
            *vote_counts.entry(vote).or_insert(0) += 1;
        }

        // Find the vote with the maximum count
        vote_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val)
            .unwrap_or(0) // Default to 0 if no votes (shouldn't happen with valid k)
    }
}