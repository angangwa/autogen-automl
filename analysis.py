"""
Wisconsin Breast Cancer Dataset Analysis
This script performs exploratory data analysis on the Wisconsin Breast Cancer dataset
and prepares it for machine learning model development.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
df = pd.read_csv('/mnt/data/wisconsin_breast_cancer.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Better column names based on domain knowledge
column_names = {
    'thickness': 'Clump Thickness',
    'size': 'Uniformity of Cell Size',
    'shape': 'Uniformity of Cell Shape',
    'adhesion': 'Marginal Adhesion',
    'single': 'Single Epithelial Cell Size',
    'nuclei': 'Bare Nuclei',
    'chromatin': 'Bland Chromatin',
    'nucleoli': 'Normal Nucleoli',
    'mitosis': 'Mitosis',
    'class': 'Diagnosis'
}

# Rename columns
df.rename(columns=column_names, inplace=True)

# Map class values to more meaningful names
df['Diagnosis_Label'] = df['Diagnosis'].map({0: 'Benign', 1: 'Malignant'})

# Handle missing values in Bare Nuclei
print(f"\nMissing values in 'Bare Nuclei': {df['Bare Nuclei'].isna().sum()}")

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
df['Bare Nuclei'] = imputer.fit_transform(df[['Bare Nuclei']])

# Check class distribution
class_counts = df['Diagnosis'].value_counts()
print("\nClass distribution:")
print(class_counts)
print("\nClass distribution percentages:")
print(class_counts / len(df) * 100)

# Create class distribution visualization
fig = px.pie(
    names=df['Diagnosis_Label'].value_counts().index,
    values=df['Diagnosis_Label'].value_counts().values,
    title='Distribution of Breast Cancer Diagnosis',
    color=df['Diagnosis_Label'].value_counts().index,
    color_discrete_map={'Benign': '#2E86C1', 'Malignant': '#E74C3C'}
)
fig.write_html('/mnt/outputs/class_distribution.html')
fig.write_image('/mnt/outputs/class_distribution.jpg')

# Feature correlation with diagnosis
feature_corr = df.drop(['id', 'Diagnosis_Label'], axis=1).corr()['Diagnosis'].sort_values(ascending=False)
print("\nFeature correlation with diagnosis:")
print(feature_corr)

# Create a correlation heatmap
correlation = df.drop(['id', 'Diagnosis_Label'], axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig('/mnt/outputs/correlation_heatmap.jpg')

# Create interactive correlation heatmap with plotly
fig = px.imshow(correlation, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                title='Correlation Heatmap of Features')
fig.write_html('/mnt/outputs/correlation_heatmap.html')
fig.write_image('/mnt/outputs/correlation_heatmap_plotly.jpg')

# Create feature distributions by class
important_features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Bare Nuclei']
for feature in important_features:
    fig = px.histogram(df, x=feature, color='Diagnosis_Label', 
                      barmode='overlay', opacity=0.7,
                      title=f'Distribution of {feature} by Diagnosis',
                      labels={feature: feature})
    fig.write_html(f'/mnt/outputs/dist_{feature.replace(" ", "_")}.html')
    fig.write_image(f'/mnt/outputs/dist_{feature.replace(" ", "_")}.jpg')

# Box plots for each feature by diagnosis
fig = px.box(
    df.melt(id_vars='Diagnosis_Label', 
            value_vars=[col for col in df.columns if col not in ['id', 'Diagnosis', 'Diagnosis_Label']]),
    x='variable', y='value', color='Diagnosis_Label',
    title='Feature Distributions by Diagnosis Class',
    labels={'variable': 'Feature', 'value': 'Value'}
)
fig.update_layout(xaxis_tickangle=-45)
fig.write_html('/mnt/outputs/boxplots.html')
fig.write_image('/mnt/outputs/boxplots.jpg')

# Prepare data for modeling
features = df.drop(['id', 'Diagnosis', 'Diagnosis_Label'], axis=1)
target = df['Diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple logistic regression model to get feature importance
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature importance:")
print(feature_importance)

# Create interactive feature importance plot
fig = px.bar(
    feature_importance, 
    x='Feature', 
    y='Importance',
    title='Feature Importance for Breast Cancer Prediction',
    labels={'Feature': 'Feature', 'Importance': 'Importance Score'},
    color='Importance'
)
fig.update_layout(xaxis_tickangle=-45)
fig.write_html('/mnt/outputs/feature_importance.html')
fig.write_image('/mnt/outputs/feature_importance.jpg')

# Calculate and plot ROC curve
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.3f})'))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Chance'))
fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    legend=dict(x=0.7, y=0.1)
)
fig.write_html('/mnt/outputs/roc_curve.html')
fig.write_image('/mnt/outputs/roc_curve.jpg')

# Create scatter plot matrix of most important features
top_features = feature_importance['Feature'].head(4).tolist()
fig = px.scatter_matrix(
    df, 
    dimensions=top_features,
    color='Diagnosis_Label',
    title='Scatter Matrix of Top 4 Important Features',
    opacity=0.7
)
fig.update_layout(height=800)
fig.write_html('/mnt/outputs/scatter_matrix.html')
fig.write_image('/mnt/outputs/scatter_matrix.jpg')

# Basic statistics by class
print("\nFeature statistics for Benign cases:")
print(df[df['Diagnosis'] == 0].describe().transpose()[['mean', 'std', 'min', 'max']])
print("\nFeature statistics for Malignant cases:")
print(df[df['Diagnosis'] == 1].describe().transpose()[['mean', 'std', 'min', 'max']])