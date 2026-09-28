# W1D4 — EDA Narrative

The India Census 2011 district-level dataset contains 640 district records and 118 columns covering population, literacy, employment, household facilities, religion, education, age groups, and purchasing-power-related indicators. The dataset contains 116 numeric columns and 2 text columns.

The descriptive statistics show substantial variation between districts. Population ranges from approximately 8,004 to 11.06 million, with a mean of approximately 1.89 million and a median of approximately 1.56 million. This difference suggests that the population distribution is not perfectly balanced and that some districts have considerably larger populations than others.

The data-quality inspection found no missing values in the dataset. No immediate missing-value treatment is therefore required. The dataset also contains no duplicate rows.

The state distribution shows that Uttar Pradesh has the highest number of district records, followed by Madhya Pradesh and Bihar. The correlation heatmap shows strong relationships between several demographic variables. This is expected because many columns represent components of larger totals, such as male and female population contributing to total population.

A potential concern is multicollinearity among highly related demographic features. Using all such features directly in a machine-learning model could introduce redundancy. Future preprocessing should therefore include feature selection or dimensionality-reduction techniques where appropriate. Additional validation should also confirm that derived columns are logically consistent with their component variables.