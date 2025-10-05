use crate::preprocessing::ProcessedPatientRecord;

pub mod logistic_regression;
pub mod naive_bayes;
pub mod knn;
pub mod decision_tree;

pub trait Model {
    fn train(&mut self, training_data: &[ProcessedPatientRecord]);
    fn predict(&self, record: &ProcessedPatientRecord) -> u8;
}

// Re-export the models for easier access
pub use logistic_regression::LogisticRegression;
pub use naive_bayes::GaussianNB;
pub use knn::KNN;
pub use decision_tree::DecisionTree;