mod preprocessing;
mod models;
mod ensemble;
mod evaluation;

use models::{
    Model,
    logistic_regression::LogisticRegression,
    naive_bayes::GaussianNB,
    knn::KNN,
    decision_tree::DecisionTree,
};
use ensemble::VotingClassifier;
use evaluation::{calculate_metrics, print_comparison_table, print_confusion_matrix, print_metrics_bar_chart, Metrics};

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

    // Create individual models
    let lr = LogisticRegression::new(0.01, 1000);
    let gnb = GaussianNB::new();
    let knn = KNN::new(5);
    let dt = DecisionTree::new(10, 2);

    // Create ensemble with all four models
    let ensemble = VotingClassifier::new(vec![
        Box::new(LogisticRegression::new(0.01, 1000)),
        Box::new(GaussianNB::new()),
        Box::new(KNN::new(5)),
        Box::new(DecisionTree::new(10, 2)),
    ]);

    let mut models: Vec<(&str, Box<dyn Model>)> = vec![
        ("Logistic Regression", Box::new(LogisticRegression::new(0.01, 1000))),
        ("Gaussian Naive Bayes", Box::new(GaussianNB::new())),
        ("KNN", Box::new(KNN::new(5))),
        ("Decision Tree", Box::new(DecisionTree::new(10, 2))),
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
    print_metrics_bar_chart(&results);

    for (name, matrix) in confusion_matrices {
        print_confusion_matrix(name, matrix);
    }
}