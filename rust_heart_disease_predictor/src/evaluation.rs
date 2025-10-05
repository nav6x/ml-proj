use crate::models::Model;
use crate::preprocessing::ProcessedPatientRecord;

#[derive(Debug)]
pub struct Metrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
}

pub fn calculate_metrics(model: &dyn Model, test_data: &[ProcessedPatientRecord]) -> (Metrics, (u32, u32, u32, u32)) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for record in test_data {
        let prediction = model.predict(record);
        match (prediction, record.target) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {},
        }
    }

    let accuracy = (tp + tn) as f32 / test_data.len() as f32;
    let precision = if (tp + fp) > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if (tp + fn_) > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
    let f1_score = if (precision + recall) > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 };

    (Metrics { accuracy, precision, recall, f1_score }, (tp, tn, fp, fn_))
}

pub fn print_comparison_table(results: &[(&str, Metrics)]) {
    println!("| {:<25} | Accuracy | Precision | Recall   | F1-Score |", "Model");
    println!("|---------------------------|----------|-----------|----------|----------|");
    for (name, metrics) in results {
        println!(
            "| {:<25} | {:.4}   | {:.4}     | {:.4}   | {:.4}   |",
            name,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score
        );
    }
    println!("|---------------------------|----------|-----------|----------|----------|");
}

