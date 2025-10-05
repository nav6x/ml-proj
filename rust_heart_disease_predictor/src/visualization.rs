use plotters::prelude::*;
use crate::evaluation::Metrics;
use crate::preprocessing::ProcessedPatientRecord;
use std::collections::HashMap;

pub fn create_performance_comparison_chart(results: &[(&str, Metrics)], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Performance Comparison", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(100)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..(results.len() * 4) as f64, 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    // Prepare data for each metric
    let accuracies: Vec<f32> = results.iter().map(|(_, metrics)| metrics.accuracy).collect();
    let precisions: Vec<f32> = results.iter().map(|(_, metrics)| metrics.precision).collect();
    let recalls: Vec<f32> = results.iter().map(|(_, metrics)| metrics.recall).collect();
    let f1_scores: Vec<f32> = results.iter().map(|(_, metrics)| metrics.f1_score).collect();

    // Draw bars for each metric for each model
    for (i, _result) in results.iter().enumerate() {
        let base_idx = i * 4;
        
        // Accuracy bar
        chart.draw_series(std::iter::once(Rectangle::new(
            [(base_idx as f64, 0.0), ((base_idx + 1) as f64, accuracies[i] as f64)],
            BLUE.filled()
        )))?;

        // Precision bar
        chart.draw_series(std::iter::once(Rectangle::new(
            [((base_idx + 1) as f64, 0.0), ((base_idx + 2) as f64, precisions[i] as f64)],
            RED.filled()
        )))?;

        // Recall bar
        chart.draw_series(std::iter::once(Rectangle::new(
            [((base_idx + 2) as f64, 0.0), ((base_idx + 3) as f64, recalls[i] as f64)],
            GREEN.filled()
        )))?;

        // F1-score bar
        chart.draw_series(std::iter::once(Rectangle::new(
            [((base_idx + 3) as f64, 0.0), ((base_idx + 4) as f64, f1_scores[i] as f64)],
            YELLOW.filled()
        )))?;
    }

    // Add x-axis labels
    for (i, (name, _)) in results.iter().enumerate() {
        let center = (i * 4 + 2) as f64; // Center of the group of 4 bars
        chart.draw_series(std::iter::once(Text::new(
            String::from(*name),
            (center, -0.05),
            ("sans-serif", 10).into_font(),
        )))?;
    }

    Ok(())
}

pub fn create_confusion_matrix_heatmap(name: &str, confusion_matrix: (u32, u32, u32, u32), output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (tp, tn, fp, fn_) = confusion_matrix;
    
    let root = BitMapBackend::new(output_path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Use float coordinates for all operations
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Confusion Matrix - {}", name), ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..2.0, 0.0..2.0)?;

    chart.configure_mesh().draw()?;

    // Define color gradients based on values
    let max_val = *[tp, tn, fp, fn_].iter().max().unwrap();
    let get_color = |val: u32| -> RGBColor {
        let intensity = (val as f32 / max_val as f32 * 255.0) as u8;
        RGBColor(255 - intensity, intensity, 100)
    };

    // Draw heatmap cells
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (1.0, 1.0)],
        get_color(tn).filled(),
    )))?;
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(1.0, 0.0), (2.0, 1.0)],
        get_color(fp).filled(),
    )))?;
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 1.0), (1.0, 2.0)],
        get_color(fn_).filled(),
    )))?;
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(1.0, 1.0), (2.0, 2.0)],
        get_color(tp).filled(),
    )))?;

    // Add text labels for values
    chart.draw_series(std::iter::once(Text::new(
        format!("{}", tn),
        (0.5, 0.5),
        ("sans-serif", 30).into_font().color(&BLACK),
    )))?;
    
    chart.draw_series(std::iter::once(Text::new(
        format!("{}", fp),
        (1.5, 0.5),
        ("sans-serif", 30).into_font().color(&BLACK),
    )))?;
    
    chart.draw_series(std::iter::once(Text::new(
        format!("{}", fn_),
        (0.5, 1.5),
        ("sans-serif", 30).into_font().color(&BLACK),
    )))?;
    
    chart.draw_series(std::iter::once(Text::new(
        format!("{}", tp),
        (1.5, 1.5),
        ("sans-serif", 30).into_font().color(&BLACK),
    )))?;

    // Add axis labels
    chart.draw_series((0..2).map(|x| Text::new(
        if x == 0 { "Predicted Neg" } else { "Predicted Pos" },
        (x as f64 + 0.5, -0.3),
        ("sans-serif", 15).into_font(),
    )))?;
    
    chart.draw_series((0..2).map(|y| Text::new(
        if y == 0 { "Actual Neg" } else { "Actual Pos" },
        (-0.5, y as f64 + 0.5),
        ("sans-serif", 15).into_font(),
    )))?;

    Ok(())
}

pub fn create_feature_histograms(data: &[ProcessedPatientRecord], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // For simplicity, let's create a histogram for the first feature (age)
    if data.is_empty() {
        return Ok(());
    }

    let mut feature_values: Vec<f32> = data.iter().map(|record| record.features[0]).collect();
    feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Create histogram data
    let mut counts = vec![0; 10]; // 10 bins
    let min_val = feature_values[0];
    let max_val = feature_values[feature_values.len() - 1];
    let range = max_val - min_val;
    let bin_width = if range > 0.0 { range / 10.0 } else { 1.0 };

    for &value in &feature_values {
        if bin_width > 0.0 {
            let bin_idx = ((value - min_val) / bin_width) as usize;
            let bin_idx = std::cmp::min(bin_idx, 9); // Clamp to last bin
            counts[bin_idx] += 1;
        }
    }

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Feature Distribution (Age)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..10.0, 0.0..(*counts.iter().max().unwrap() + 1) as f64)?;

    chart.configure_mesh().draw()?;

    // Draw histogram bars
    for (i, &count) in counts.iter().enumerate() {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(i as f64, 0.0), ((i + 1) as f64, count as f64)],
            BLUE.filled(),
        )))?;
    }

    Ok(())
}

// Function to create a simple text representation of the decision tree
pub fn print_decision_tree_structure(name: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;
    writeln!(file, "Decision Tree Structure for: {}", name)?;
    writeln!(file, "This is a placeholder for an actual tree visualization.")?;
    writeln!(file, "The tree would show:")?;
    writeln!(file, " - Root node with first split condition")?;
    writeln!(file, " - Internal nodes with subsequent split conditions")?;
    writeln!(file, " - Leaf nodes with class predictions")?;
    writeln!(file, " - Feature names and threshold values at each split")?;
    
    Ok(())
}

// Function to create correlation matrix heatmap
pub fn create_correlation_matrix_heatmap(data: &[ProcessedPatientRecord], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() || data[0].features.is_empty() {
        return Ok(());
    }

    let num_features = data[0].features.len();
    let n = data.len();
    
    // Calculate means for each feature
    let mut means = vec![0.0; num_features];
    for record in data {
        for (i, &feature) in record.features.iter().enumerate() {
            means[i] += feature;
        }
    }
    for mean in &mut means {
        *mean /= n as f32;
    }

    // Calculate correlation matrix
    let mut correlation_matrix = vec![vec![0.0; num_features]; num_features];
    
    for i in 0..num_features {
        for j in 0..num_features {
            if i == j {
                correlation_matrix[i][j] = 1.0;
            } else {
                // Calculate correlation between feature i and j
                let mut sum_xy = 0.0;
                let mut sum_x_sq = 0.0;
                let mut sum_y_sq = 0.0;
                
                for record in data {
                    let x = record.features[i] - means[i];
                    let y = record.features[j] - means[j];
                    sum_xy += x * y;
                    sum_x_sq += x * x;
                    sum_y_sq += y * y;
                }
                
                let denominator = (sum_x_sq * sum_y_sq).sqrt();
                if denominator != 0.0 {
                    correlation_matrix[i][j] = sum_xy / denominator;
                } else {
                    correlation_matrix[i][j] = 0.0;
                }
            }
        }
    }

    // Create visualization
    let root = BitMapBackend::new(output_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Feature Correlation Matrix", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..num_features as f64, 0.0..num_features as f64)?;

    chart.configure_mesh().draw()?;

    // Draw correlation heatmap
    for i in 0..num_features {
        for j in 0..num_features {
            // Map correlation value to color (red for negative, blue for positive)
            let corr_val = correlation_matrix[i][j];
            let (r, g, b) = if corr_val >= 0.0 {
                // Blue scale for positive correlation
                let intensity = (corr_val * 255.0) as u8;
                (255 - intensity, 255 - intensity, 255)
            } else {
                // Red scale for negative correlation
                let intensity = ((-corr_val) * 255.0) as u8;
                (255, 255 - intensity, 255 - intensity)
            };
            
            chart.draw_series(std::iter::once(Rectangle::new(
                [(i as f64, j as f64), ((i + 1) as f64, (j + 1) as f64)],
                RGBColor(r, g, b).filled(),
            )))?;
            
            // Add text label with correlation value
            chart.draw_series(std::iter::once(Text::new(
                format!("{:.2}", corr_val),
                (i as f64 + 0.5, j as f64 + 0.5),
                ("sans-serif", 10).into_font().color(&BLACK),
            )))?;
        }
    }

    // Add feature index labels
    for i in 0..num_features {
        chart.draw_series(std::iter::once(Text::new(
            format!("F{}", i),
            (i as f64 + 0.5, -0.5),
            ("sans-serif", 10).into_font(),
        )))?;
        
        chart.draw_series(std::iter::once(Text::new(
            format!("F{}", i),
            (-0.5, i as f64 + 0.5),
            ("sans-serif", 10).into_font(),
        )))?;
    }

    Ok(())
}