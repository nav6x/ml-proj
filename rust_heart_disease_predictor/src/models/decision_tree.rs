use crate::preprocessing::ProcessedPatientRecord;

#[derive(Debug, Clone)]
pub enum Node {
    Leaf(u8),
    Internal {
        feature_index: usize,
        threshold: f32,
        left: Box<Node>,
        right: Box<Node>,
    },
}

pub struct DecisionTree {
    root: Option<Node>,
    max_depth: usize,
    min_samples_split: usize,
}

impl super::Model for DecisionTree {
    fn train(&mut self, training_data: &[ProcessedPatientRecord]) {
        if !training_data.is_empty() {
            self.root = Some(self.build_tree(training_data, 0));
        }
    }

    fn predict(&self, record: &ProcessedPatientRecord) -> u8 {
        match &self.root {
            Some(node) => self.predict_from_node(node, &record.features),
            None => 0, // Default prediction if tree wasn't built
        }
    }
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTree {
            root: None,
            max_depth,
            min_samples_split,
        }
    }


    fn build_tree(&self, data: &[ProcessedPatientRecord], depth: usize) -> Node {
        // Check stopping conditions
        if data.is_empty() {
            return Node::Leaf(0);
        }

        // Check if all samples have the same target
        let first_target = data[0].target;
        if data.iter().all(|record| record.target == first_target) {
            return Node::Leaf(first_target);
        }

        // Check stopping conditions: max depth or minimum samples
        if depth >= self.max_depth || data.len() < self.min_samples_split {
            return Node::Leaf(self.most_common_class(data));
        }

        // Find the best split
        if let Some((best_feature, best_threshold)) = self.find_best_split(data) {
            let (left_data, right_data) = self.split_data(data, best_feature, best_threshold);

            let left_node = Box::new(self.build_tree(&left_data, depth + 1));
            let right_node = Box::new(self.build_tree(&right_data, depth + 1));

            Node::Internal {
                feature_index: best_feature,
                threshold: best_threshold,
                left: left_node,
                right: right_node,
            }
        } else {
            // If no good split is found, create a leaf with the majority class
            Node::Leaf(self.most_common_class(data))
        }
    }

    fn find_best_split(&self, data: &[ProcessedPatientRecord]) -> Option<(usize, f32)> {
        if data.is_empty() {
            return None;
        }

        let num_features = data[0].features.len();
        let mut best_gini = f32::MAX;
        let mut best_split: Option<(usize, f32)> = None;

        for feature_idx in 0..num_features {
            // Get all unique values for this feature
            let mut feature_values: Vec<f32> = data
                .iter()
                .map(|record| record.features[feature_idx])
                .collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            feature_values.dedup();

            for i in 0..feature_values.len() - 1 {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                let (left_data, right_data) = self.split_data(data, feature_idx, threshold);
                
                if left_data.is_empty() || right_data.is_empty() {
                    continue;
                }

                let gini = self.calculate_split_gini(&left_data, &right_data);
                
                if gini < best_gini {
                    best_gini = gini;
                    best_split = Some((feature_idx, threshold));
                }
            }
        }

        best_split
    }

    fn split_data(
        &self,
        data: &[ProcessedPatientRecord],
        feature_idx: usize,
        threshold: f32,
    ) -> (Vec<ProcessedPatientRecord>, Vec<ProcessedPatientRecord>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for record in data {
            if record.features[feature_idx] <= threshold {
                left.push(record.clone());
            } else {
                right.push(record.clone());
            }
        }

        (left, right)
    }

    fn calculate_split_gini(&self, left_data: &[ProcessedPatientRecord], right_data: &[ProcessedPatientRecord]) -> f32 {
        let total_size = (left_data.len() + right_data.len()) as f32;

        let left_gini = self.calculate_gini(left_data);
        let right_gini = self.calculate_gini(right_data);

        let weighted_gini = (left_data.len() as f32 / total_size) * left_gini
            + (right_data.len() as f32 / total_size) * right_gini;

        weighted_gini
    }

    fn calculate_gini(&self, data: &[ProcessedPatientRecord]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let mut class_counts = std::collections::HashMap::new();
        for record in data {
            *class_counts.entry(record.target).or_insert(0) += 1;
        }

        let total = data.len() as f32;
        let mut gini = 1.0;

        for &count in class_counts.values() {
            let proportion = count as f32 / total;
            gini -= proportion * proportion;
        }

        gini
    }

    fn most_common_class(&self, data: &[ProcessedPatientRecord]) -> u8 {
        let mut class_counts = std::collections::HashMap::new();
        for record in data {
            *class_counts.entry(record.target).or_insert(0) += 1;
        }

        class_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(class, _)| class)
            .unwrap_or(0)
    }

    fn predict_from_node(&self, node: &Node, features: &[f32]) -> u8 {
        match node {
            Node::Leaf(class) => *class,
            Node::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if features[*feature_index] <= *threshold {
                    self.predict_from_node(left, features)
                } else {
                    self.predict_from_node(right, features)
                }
            }
        }
    }
}
