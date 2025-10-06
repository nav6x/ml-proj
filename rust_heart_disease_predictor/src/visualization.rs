use plotters::prelude::*;
use crate::evaluation::Metrics;
use crate::preprocessing::ProcessedPatientRecord;

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

    root.present()?;
    Ok(())
}

pub fn save_performance_chart(results: &[(&str, Metrics)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("performance_chart.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(50) // Increased x-label area
        .y_label_area_size(60) // Increased y-label area
        .margin(10) // Increased margin
        .caption("Model Performance Comparison", ("sans-serif", 40.0).into_font()) // Adjusted font size
        .build_cartesian_2d(
            0f64..4f64,
            0f32..1f32,
        )?;

    chart
        .configure_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_label_style(("sans-serif", 15.0).into_font()) // Y-label font size
        .x_label_style(("sans-serif", 15.0).into_font()) // X-label font size
        .y_desc("Score")
        .x_desc("Metrics") // Added X-axis description
        .axis_style(&BLACK.mix(0.1)) // Subtle Y-axis grid lines
        .draw()?;

    let colors = [
        BLUE.mix(0.8), 
        GREEN.mix(0.8), 
        MAGENTA.mix(0.8),
        RED.mix(0.8),
        CYAN.mix(0.8),
    ];
    results.iter().enumerate().for_each(|(i, (name, metrics))| {
        let data = [
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
        ];

        let num_models = results.len() as f64;
        let bar_width_ratio = 0.7 / num_models; // Ratio of the segment width each bar takes
        let bar_padding_ratio = (1.0 - 0.7) / (num_models + 1.0); // Padding between bars

        // Draw the bars for the current model
        chart.draw_series(
            data.iter().enumerate().map(|(j, &v)| {
                let x_start = j as f64 + bar_padding_ratio + i as f64 * (bar_width_ratio + bar_padding_ratio);
                let x_end = x_start + bar_width_ratio;
                let color = colors[i].filled();

                Rectangle::new(
                    [(x_start, 0.0f32), (x_end, v)],
                    color,
                )
            })
        )
        .unwrap()
        .label(*name) // Add label for the legend
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i].filled())); // Draw a colored rectangle in the legend

        // Draw data labels for the current model
        chart.draw_series(
            data.iter().enumerate().map(|(j, &v)| {
                let x_start = j as f64 + bar_padding_ratio + i as f64 * (bar_width_ratio + bar_padding_ratio);
                let x_end = x_start + bar_width_ratio;

                let label_text = format!("{:.2}", v);
                let label_point = ((x_start + x_end) / 2.0, v + 0.02); // Position slightly above the bar
                Text::new(label_text, label_point, ("sans-serif", 12.0).into_font())
            })
        ).unwrap();
    });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft) // Position legend
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    // Manually add a legend title since legend_text is not available
    root.draw_text(
        "Models",
        &("sans-serif", 20.0).into_font().color(&BLACK),
        (50, 50) // Adjust position as needed
    )?;

    root.present()?;
    println!("Saved performance chart to performance_chart.png");

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
pub fn save_confusion_matrix(
    name: &str,
    confusion_matrix: (u32, u32, u32, u32),
) -> Result<(), Box<dyn std::error::Error>> {
    let (tp, tn, fp, fn_) = confusion_matrix;
    let file_name = format!("confusion_matrix_{}.png", name.to_lowercase().replace(" ", "_"));
    let root = BitMapBackend::new(&file_name, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Confusion Matrix: {}", name), ("sans-serif", 30.0).into_font())
        .x_label_area_size(50) // Increased space for x-axis label
        .y_label_area_size(50) // Increased space for y-axis label
        .margin(10)
        .build_cartesian_2d(0..2, 0..2)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Predicted") // X-axis label
        .y_desc("Actual") // Y-axis label
        .x_label_style(("sans-serif", 20.0).into_font())
        .y_label_style(("sans-serif", 20.0).into_font())
        .draw()?;

    let max_val = *[tp, tn, fp, fn_].iter().max().unwrap_or(&1) as f32;

    let cells = [
        (0, 1, tn, "True Negative"), 
        (1, 1, fp, "False Positive"), 
        (0, 0, fn_, "False Negative"), 
        (1, 0, tp, "True Positive")
    ];

    for (x, y, val, label) in cells.iter() {
        // Use a gradient from white to blue based on the value
        let color_intensity = *val as f32 / max_val;
        let cell_color = BLUE.mix(color_intensity as f64);

        // Draw the filled cell rectangle
        chart.draw_series(std::iter::once(
            Rectangle::new([(*x, *y), (*x + 1, *y + 1)], cell_color.filled())
        ))?;

        // Draw the border rectangle
        chart.draw_series(std::iter::once(
            Rectangle::new([(*x, *y), (*x + 1, *y + 1)], ShapeStyle::from(&BLACK).stroke_width(2))
        ))?;

        // Draw the value text in the center of the cell
        chart.draw_series(std::iter::once(
            Text::new(
                format!("{}\n{}", label, val),
                ((*x as f64 + 0.5) as i32, (*y as f64 + 0.5) as i32), // Center of the cell, cast to i32
                ("sans-serif", 25.0).into_font().color(&BLACK)
            )
        ))?;
    }

    root.present()?;
    println!("Saved confusion matrix to {}", file_name);

    Ok(())
}
