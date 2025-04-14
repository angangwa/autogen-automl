# Wisconsin Breast Cancer Dataset Description

## Overview
The Wisconsin Breast Cancer dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image and are used to classify breast masses as either benign or malignant.

## Dataset Statistics
- **Total samples**: 699 instances
- **Features**: 9 cytological characteristics (plus ID column)
- **Target variable**: Diagnosis (0 = Benign, 1 = Malignant)
- **Class distribution**: 458 Benign (65.5%), 241 Malignant (34.5%)
- **Missing values**: 16 missing values in the 'Bare Nuclei' feature (imputed with median)

![Class Distribution](class_distribution.jpg)

## Features Description
Each feature is measured on a scale of 1-10, with 1 being the closest to benign characteristics and 10 being the most anaplastic (abnormal, indicative of malignancy):

1. **Clump Thickness**: Assesses if cells are mono or multilayered
2. **Uniformity of Cell Size**: Evaluates the consistency in cell size
3. **Uniformity of Cell Shape**: Evaluates the consistency in cell shape
4. **Marginal Adhesion**: Measures loss of adhesion at cell boundaries
5. **Single Epithelial Cell Size**: Relates to cell uniformity
6. **Bare Nuclei**: Refers to nuclei not surrounded by cytoplasm
7. **Bland Chromatin**: Describes the texture of the nucleus
8. **Normal Nucleoli**: Measures the prominence of nucleoli
9. **Mitosis**: Counts mitotic figures (cell division rate)

## Feature Importance
Analysis revealed that certain features have stronger predictive power for cancer diagnosis:

1. **Bare Nuclei** (Importance: 1.52)
2. **Clump Thickness** (Importance: 1.28)
3. **Uniformity of Cell Shape** (Importance: 0.96)
4. **Bland Chromatin** (Importance: 0.93)
5. **Mitosis** (Importance: 0.56)

![Feature Importance](feature_importance.jpg)

## Feature Correlations
The correlation analysis shows strong relationships between several features and the diagnosis:

- 'Bare Nuclei' has the highest correlation with diagnosis (0.82)
- 'Uniformity of Cell Shape' and 'Uniformity of Cell Size' also show strong correlations (0.82 and 0.82 respectively)
- 'Bland Chromatin' has a correlation of 0.76 with diagnosis
- 'Mitosis' has the lowest correlation (0.42), but still significant

![Correlation Heatmap](correlation_heatmap_plotly.jpg)

## Class Separation
The top features show clear separation between benign and malignant cases:

- **Benign masses** typically have lower values across features:
  - Average Clump Thickness: 2.96
  - Average Bare Nuclei: 1.34
  - Average Uniformity of Cell Shape: 1.44

- **Malignant masses** show higher values:
  - Average Clump Thickness: 7.20
  - Average Bare Nuclei: 7.57
  - Average Uniformity of Cell Shape: 6.56

## Model Performance Indication
A preliminary logistic regression model shows excellent performance with an AUC of 0.998, indicating strong predictive capability of these features.

![ROC Curve](roc_curve.jpg)

## Visualization of Key Features
The scatter matrix of the top features shows clear clustering of benign and malignant cases, confirming their strong discriminatory power.

![Scatter Matrix](scatter_matrix.jpg)

## Recommendations for ML Approach
1. **Feature Selection**: Focus on the top 5 features for a parsimonious model
2. **Algorithms to Consider**: 
   - Logistic Regression (good baseline with interpretability)
   - Random Forest or Gradient Boosting (likely to perform well)
   - Support Vector Machines (effective for this type of classification)
3. **Evaluation Metrics**: Prioritize sensitivity over specificity; use F1-score, AUC, and confusion matrix
4. **Handling Class Imbalance**: Consider class weighting or SMOTE for addressing the 65:35 ratio
5. **Model Interpretability**: Maintain feature explanations for clinical context
6. **Preprocessing**: Feature scaling is advisable given the varying ranges of feature values