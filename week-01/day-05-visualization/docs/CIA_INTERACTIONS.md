# W1D5 CIA Interactions — Data Visualization

## Interaction 1 — Full-Stack Mentor Code Review

### Mentor Focus
Reviewed the completed W1D5 Matplotlib and Seaborn data visualization implementation against the internship requirements.

### Relevant Feedback
- The implementation correctly loads the India Census 2011 dataset using `pathlib.Path` and validates that the dataset exists.
- Numeric columns are selected using `select_dtypes(include="number")`.
- A population distribution histogram is generated and saved as `population_distribution.png`.
- A population box plot is generated using Seaborn and saved as `population_boxplot.png`.
- A correlation heatmap is generated using selected numeric features and saved as `correlation_heatmap.png`.
- A top-10 states bar chart is generated using state district counts and saved as `top_10_states.png`.
- The output directory is created with `mkdir(parents=True, exist_ok=True)`.
- Figures are closed after saving using `plt.close()`.
- The implementation uses separate functions, docstrings, and a `__main__` guard.
- CIA identified the use of `Path(__file__).resolve().parents[3]` as a maintainability consideration because the path depends on the current project structure.
- CIA also suggested validating expected column names and reusing or removing the numeric-column variable.
- Additional suggestions included annotating the heatmap and improving plot reproducibility.

### Action Taken
The mentor review confirmed that the W1D5 implementation covers the required visualization workflow. The path, column validation, heatmap annotation, and reproducibility suggestions were treated as maintainability improvements rather than reasons to expand the internship task beyond its required scope.

---

## Interaction 2 — Full-Stack Mentor Deliverables Review

### Mentor Focus
Reviewed the W1D5 data visualization workflow and its deliverables against the expected internship requirements.

### Relevant Feedback
- The implementation correctly uses Matplotlib and Seaborn for the required visualizations.
- The population histogram and box plot provide distribution and outlier views.
- The correlation heatmap provides a visual overview of relationships between numeric features.
- The top-10 states bar chart provides a categorical summary of district counts.
- Visualization files are saved in a deterministic `outputs` directory.
- The implementation uses `pathlib` for file handling and closes generated figures.
- CIA identified the fixed project-root path and hard-coded column names as potential maintainability concerns.
- CIA also noted that selecting the first 12 numeric columns for the correlation heatmap should either be documented or changed to use all numeric columns, depending on the intended visualization scope.
- These suggestions were considered production-style improvements and did not change the core W1D5 requirements.

### Action Taken
The W1D5 deliverables were kept focused on the assigned data visualization requirements. No unrelated features or unnecessary production-level changes were introduced.

---

## Summary

The CIA review confirmed that the W1D5 implementation successfully covers the required data visualization workflow using Matplotlib and Seaborn.

The completed workflow includes a population distribution histogram, population box plot, correlation heatmap, and top-10 state bar chart, with all visualization outputs saved for evidence.

The mentor also identified maintainability and reproducibility improvements, particularly around path resolution, column validation, and correlation-column selection. These were documented as improvement opportunities rather than unnecessary scope changes.