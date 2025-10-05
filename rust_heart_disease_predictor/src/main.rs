mod preprocessing;
mod models;
mod ensemble;
mod evaluation;
mod visualization;

use models::{
    Model,
    logistic_regression::LogisticRegression,
    naive_bayes::GaussianNB,
};
use ensemble::VotingClassifier;
use evaluation::{calculate_metrics, print_comparison_table};
use visualization::{save_performance_chart, save_confusion_matrix};

fn main() {
    println!("Rust Heart Disease Predictor");

    // Load and preprocess data
    let data_path = "data/processed.cleveland.data";
    let mut records = match preprocessing::load_and_preprocess_data(data_path) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("Error loading data: {}", e);
            return;
        }
    };

    // Split data
    let (train_set, test_set) = preprocessing::train_test_split(&mut records, 0.2);

    // Create models
    let lr = LogisticRegression::new(0.01, 1000);
    let gnb = GaussianNB::new();

    let ensemble = VotingClassifier::new(vec![Box::new(lr), Box::new(gnb)]);

    let mut models: Vec<(&str, Box<dyn Model>)> = vec![
        ("Logistic Regression", Box::new(LogisticRegression::new(0.01, 1000))),
        ("Gaussian Naive Bayes", Box::new(GaussianNB::new())),
        ("Voting Classifier", Box::new(ensemble)),
    ];

    let mut results = Vec::new();
    let mut confusion_matrices = Vec::new();

    for (name, model) in &mut models {
        model.train(&train_set);
        let (metrics, confusion_matrix) = calculate_metrics(model.as_ref(), &test_set);
        results.push((*name, metrics));
        confusion_matrices.push((*name, confusion_matrix));
    }

    print_comparison_table(&results);

    if let Err(e) = save_performance_chart(&results) {
        eprintln!("Error saving performance chart: {}", e);
    }

    for (name, matrix) in confusion_matrices {
        if let Err(e) = save_confusion_matrix(name, matrix) {
            eprintln!("Error saving confusion matrix for {}: {}", name, e);
        }
    }
}
