use crate::models::Model;
use crate::preprocessing::ProcessedPatientRecord;

pub struct VotingClassifier {
    models: Vec<Box<dyn Model>>,
}

impl VotingClassifier {
    pub fn new(models: Vec<Box<dyn Model>>) -> Self {
        VotingClassifier { models }
    }
}

impl Model for VotingClassifier {
    fn train(&mut self, training_data: &[ProcessedPatientRecord]) {
        for model in &mut self.models {
            model.train(training_data);
        }
    }

    fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        let mut votes = Vec::new();
        for model in &self.models {
            votes.push(model.predict(record));
        }

        let mut vote_counts = std::collections::HashMap::new();
        for vote in votes.iter() {
            *vote_counts.entry(vote).or_insert(0) += 1;
        }

        // Find the vote with the maximum count.
        // In case of a tie, the first model's prediction is effectively chosen.
        vote_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| *val)
            .unwrap_or_else(|| *votes.first().unwrap_or(&0))
    }
}
