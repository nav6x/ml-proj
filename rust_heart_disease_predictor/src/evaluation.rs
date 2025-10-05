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

pub fn print_confusion_matrix(name: &str, confusion_matrix: (u32, u32, u32, u32)) {
    let (tp, tn, fp, fn_) = confusion_matrix;
    println!("\nConfusion Matrix for: {}", name);
    println!("-------------------------");
    println!("|          | Predicted |");
    println!("|          | Neg | Pos |");
    println!("|----------|-----|-----|");
    println!("| Actual N | {:<3} | {:<3} |", tn, fp);
    println!("| Actual P | {:<3} | {:<3} |", fn_, tp);
    println!("-------------------------");
}

pub fn print_metrics_bar_chart(results: &[(&str, Metrics)]) {
    println!("\nPerformance Metrics Bar Chart:");
    let metrics = ["Accuracy", "Precision", "Recall", "F1-Score"];
    for metric_name in metrics.iter() {
        println!("\n{}:", metric_name);
        for (name, metrics_values) in results {
            let value = match *metric_name {
                "Accuracy" => metrics_values.accuracy,
                "Precision" => metrics_values.precision,
                "Recall" => metrics_values.recall,
                "F1-Score" => metrics_values.f1_score,
                _ => 0.0,
            };
            let bar = "█".repeat((value * 50.0) as usize);
            println!("{:<25} |{:<50}| {:.4}", name, bar, value);
        }
    }
}

