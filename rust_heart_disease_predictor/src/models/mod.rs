use crate::preprocessing::ProcessedPatientRecord;

pub mod logistic_regression;
pub mod naive_bayes;
pub mod knn;
pub mod decision_tree;

pub trait Model {
    fn train(&mut self, training_data: &[ProcessedPatientRecord]);
    fn predict(&self, record: &ProcessedPatientRecord) -> u8;
}