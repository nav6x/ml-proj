use crate::evaluation::Metrics;
use plotters::prelude::*;

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

    let colors = [BLUE.mix(0.8), GREEN.mix(0.8), MAGENTA.mix(0.8)];
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
