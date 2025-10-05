### **Project: Rust Heart Disease Predictor**

*   **Goal:** Create a binary application that trains four distinct ML models, combines their predictions, and evaluates their performance.
*   **Language:** Rust
*   **Core Principle:** Implement ML algorithms from first principles to minimize external dependencies, relying primarily on the Rust standard library and basic crates for tasks like CSV parsing and data structures.

---

### **Phase 1: Foundation & Data Preparation**

1.  **Project Setup:**
    *   Initialize a new Rust binary project: `cargo new rust_heart_disease_predictor`
    *   Create a directory structure:
        ```
        .
        ├── Cargo.toml
        ├── data/
        └── src/
            ├── main.rs
            ├── lib.rs
            ├── error.rs
            ├── preprocessing.rs
            └── models/
                ├── mod.rs
                ├── logistic_regression.rs
                ├── knn.rs
                ├── naive_bayes.rs
                └── decision_tree.rs
        ```

2.  **Data Acquisition:**
    *   Download the processed Cleveland dataset from the UCI repository. I will add a step to fetch it and place it in the `data/` directory.
    *   I will use this URL: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`

3.  **Data Parsing & Preprocessing (`preprocessing.rs`):**
    *   Add the `csv` crate for parsing the data file.
    *   Implement a robust parser that handles the comma-delimited format and missing values (represented by '?').
    *   Define a `PatientRecord` struct to hold the 14 attributes (age, sex, cp, etc.).
    *   Implement data cleaning:
        *   Decide on a strategy for missing values (e.g., imputation with the mean/median or row removal).
        *   Convert the multi-class `num` (diagnosis) field into a binary classification target (0 for no disease, 1 for presence of disease).
    *   Implement a function to split the dataset into training and testing sets (e.g., an 80/20 split).

---

### **Phase 2: Individual Model Implementation (`models/*.rs`)**

*For each model, the goal is to write the algorithm from scratch, using only basic math operations. No high-level ML crates like `linfa` or `tch`.*

1.  **Model 1: Logistic Regression:**
    *   Implement the sigmoid function.
    *   Write a `train` function that uses gradient descent to learn the model weights. It will take the training data and hyperparameters (learning rate, number of iterations) as input.
    *   Write a `predict` function that takes a patient record and the learned weights to produce a probability and a binary prediction.

2.  **Model 2: K-Nearest Neighbors (KNN):**
    *   Implement a function to calculate Euclidean distance between two patient records.
    *   The `predict` function will find the `k` nearest neighbors from the training set for a given test record.
    *   The final prediction will be the majority class among the neighbors. The `train` function for KNN is trivial; it just stores the training data.

3.  **Model 3: Gaussian Naive Bayes:**
    *   The `train` function will calculate the mean, variance, and prior probability for each class (0 and 1) from the training data.
    *   The `predict` function will use the Gaussian probability density function to calculate the likelihood of the given features for each class.
    *   It will then apply Bayes' theorem to determine the most probable class.

4.  **Model 4: Decision Tree (C4.5 or CART variant):**
    *   This is the most complex model to implement.
    *   Define `Node` and `Tree` structures.
    *   Implement a function to calculate Gini impurity or entropy for a set of data.
    *   Write a recursive `build_tree` function that finds the best feature and threshold to split the data at each node, aiming to minimize impurity.
    *   Implement a `predict` function that traverses the tree with a given patient record to arrive at a leaf node and return its prediction.

---

### **Phase 3: Integration & Ensemble Model**

1.  **Model Trait (`models/mod.rs`):**
    *   Define a common `Model` trait that all four structs will implement.
    *   ```rust
      pub trait Model {
          fn train(&mut self, training_data: &[PatientRecord]);
          fn predict(&self, record: &PatientRecord) -> u8;
      }
      ```

2.  **Ensemble Implementation (`main.rs` or `ensemble.rs`):**
    *   Create a `VotingClassifier` struct that holds an instance of each of the four models.
    *   The `train` method will call the respective `train` method for each internal model.
    *   The `predict` method will call `predict` on each model and return the majority vote. For a 4-model ensemble, a tie-breaking rule will be needed (e.g., default to a specific class or use the prediction from a designated "default" model).

---

### **Phase 4: Application & Verification**

1.  **Main Application Logic (`main.rs`):**
    *   Load and preprocess the data.
    *   Instantiate and train all four models plus the `VotingClassifier`.
    *   Iterate through the test set.
    *   For each record in the test set, generate a prediction from each of the 5 models (4 individual + 1 ensemble).

2.  **Performance Evaluation:**
    *   Implement functions to calculate and report performance metrics for each model:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-Score
    *   Print a comparison table to the console showing how each model and the final ensemble performed on the test data.

---

### **Phase 5: Visualization**

1.  **Individual Model Performance:**
    *   Generate bar charts to compare the key performance metrics (Accuracy, Precision, Recall, F1-Score) across the four individual models.
    *   For Logistic Regression, plot the decision boundary if possible (may require dimensionality reduction to 2D).
    *   For KNN, create a scatter plot of the data (using 2D PCA) and highlight the neighbors for a few example test points.
    *   For the Decision Tree, implement a text-based or graphical representation of the final tree structure.

2.  **Ensemble Model Performance:**
    *   Add the `VotingClassifier`'s performance to the comparison bar charts.
    *   Create a confusion matrix visualization for the ensemble model to show True Positives, True Negatives, False Positives, and False Negatives.

3.  **Data Visualization:**
    *   Generate histograms for each of the 13 features to understand their distributions.
    *   Create a correlation matrix heatmap to visualize the relationships between different features.
