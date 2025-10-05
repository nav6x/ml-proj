use crate::preprocessing::ProcessedPatientRecord;

pub struct LogisticRegression {
    weights: Vec<f32>,
    learning_rate: f32,
    epochs: usize,
}

impl super::Model for LogisticRegression {
    fn train(&mut self, data: &[ProcessedPatientRecord]) {
        if data.is_empty() {
            return;
        }
        let num_features = data[0].features.len();
        // Initialize weights with zeros, including bias term
        self.weights = vec![0.0; num_features + 1];

        for _ in 0..self.epochs {
            for record in data {
                let mut features_with_bias = record.features.clone();
                features_with_bias.insert(0, 1.0); // Bias term

                let z = features_with_bias
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(f, w)| f * w)
                    .sum();

                let prediction = Self::sigmoid(z);
                let error = record.target as f32 - prediction;

                for i in 0..self.weights.len() {
                    self.weights[i] += self.learning_rate * error * features_with_bias[i];
                }
            }
        }
    }

    fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        let mut features_with_bias = record.features.clone();
        features_with_bias.insert(0, 1.0); // Bias term

        let z = features_with_bias
            .iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        let probability = Self::sigmoid(z);
        if probability >= 0.5 {
            1
        } else {
            0
        }
    }
}

impl LogisticRegression {
    pub fn new(learning_rate: f32, epochs: usize) -> Self {
        LogisticRegression {
            weights: Vec::new(),
            learning_rate,
            epochs,
        }
    }

    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }
}
