"""
================================================================================
PROJECT: DETECTING ANOMALIES IN HEALTHCARE INSURANCE CLAIMS
================================================================================

Project Overview:
-----------------
This project aims to detect fraudulent healthcare insurance claims using 
machine learning techniques. We will analyze a dataset of 100,000 insurance 
claims and build models to identify suspicious patterns.

Learning Objectives:
--------------------
1. Understanding the complete data science workflow
2. Exploratory Data Analysis (EDA)
3. Data preprocessing and feature engineering
4. Building and comparing multiple machine learning models
5. Evaluating model performance
6. Interpreting results for business insights

Author: Beginner Data Science Project
Level: Beginner
Date: 2024
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ============================================================================
# Libraries are pre-written code that provide useful functions
# We import them at the beginning so we can use them throughout our project

# Data manipulation libraries
import pandas as pd  # For working with tabular data (like Excel spreadsheets)
import numpy as np   # For numerical operations and arrays

# Data visualization libraries
import matplotlib.pyplot as plt  # For creating static plots and charts
import seaborn as sns           # For creating beautiful statistical visualizations

# Machine learning libraries from scikit-learn
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For data preprocessing
from sklearn.ensemble import RandomForestClassifier, IsolationForest  # Machine learning algorithms
from sklearn.linear_model import LogisticRegression  # Another ML algorithm
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)  # For evaluating model performance

# For handling imbalanced datasets
from imblearn.over_sampling import SMOTE  # Synthetic Minority Over-sampling Technique

# Utility libraries
import warnings  # To suppress warning messages
warnings.filterwarnings('ignore')  # Ignore warnings to keep output clean

# Set random seed for reproducibility
# This ensures we get the same results every time we run the code
np.random.seed(42)

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')  # Set a nice visual style for our plots
sns.set_palette("husl")  # Set color palette for seaborn plots

print("="*80)
print("HEALTHCARE INSURANCE FRAUD DETECTION PROJECT")
print("="*80)
print("\n✓ All libraries imported successfully!")
print("✓ Random seed set for reproducibility (seed=42)")
print("✓ Visualization settings configured\n")


# ============================================================================
# SECTION 2: LOADING AND EXPLORING THE DATA
# ============================================================================
# The first step in any data science project is to load and understand your data

print("="*80)
print("STEP 1: LOADING THE DATA")
print("="*80)

# Load the CSV file into a pandas DataFrame
# A DataFrame is like a table in Excel - it has rows and columns
data = pd.read_csv('/mnt/user-data/uploads/Claude_healthcare_insurance_claims_100k.csv')

print(f"\n✓ Data loaded successfully!")
print(f"✓ Total number of claims: {len(data):,}")
print(f"✓ Total number of features: {len(data.columns)}")

# Display basic information about the dataset
print("\n" + "-"*80)
print("DATASET OVERVIEW")
print("-"*80)
print(f"\nDataset Shape: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"\nFirst few rows of the dataset:")
print(data.head())

# Display column names and their data types
print("\n" + "-"*80)
print("COLUMN INFORMATION")
print("-"*80)
print("\nColumn Names and Data Types:")
print(data.dtypes)

# Check for missing values
# Missing values are empty cells in our data that need to be handled
print("\n" + "-"*80)
print("MISSING VALUES CHECK")
print("-"*80)
missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found! ✓")

# Display basic statistics for numerical columns
# This shows us the distribution of our numerical features
print("\n" + "-"*80)
print("STATISTICAL SUMMARY (NUMERICAL FEATURES)")
print("-"*80)
print(data.describe())


# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# EDA helps us understand patterns, relationships, and anomalies in our data

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# 3.1: Analyze the target variable (IsFraud)
# The target variable is what we want to predict (fraudulent or not)
print("\n" + "-"*80)
print("TARGET VARIABLE ANALYSIS: IsFraud")
print("-"*80)

fraud_counts = data['IsFraud'].value_counts()
fraud_percentages = data['IsFraud'].value_counts(normalize=True) * 100

print(f"\nFraud Distribution:")
print(f"  Legitimate Claims (0): {fraud_counts[0]:,} ({fraud_percentages[0]:.2f}%)")
print(f"  Fraudulent Claims (1): {fraud_counts[1]:,} ({fraud_percentages[1]:.2f}%)")
print(f"\n  Class Imbalance Ratio: {fraud_counts[0]/fraud_counts[1]:.2f}:1")
print(f"  (For every 1 fraudulent claim, there are {fraud_counts[0]/fraud_counts[1]:.1f} legitimate claims)")

# 3.2: Analyze fraud types
print("\n" + "-"*80)
print("FRAUD TYPE DISTRIBUTION")
print("-"*80)
fraud_type_dist = data['FraudType'].value_counts()
print("\nTypes of fraud detected:")
print(fraud_type_dist)

# 3.3: Analyze claim amounts
print("\n" + "-"*80)
print("CLAIM AMOUNT ANALYSIS")
print("-"*80)
print(f"\nClaim Amount Statistics:")
print(f"  Mean Claim Amount: ${data['ClaimAmount'].mean():,.2f}")
print(f"  Median Claim Amount: ${data['ClaimAmount'].median():,.2f}")
print(f"  Minimum Claim Amount: ${data['ClaimAmount'].min():,.2f}")
print(f"  Maximum Claim Amount: ${data['ClaimAmount'].max():,.2f}")
print(f"  Standard Deviation: ${data['ClaimAmount'].std():,.2f}")

# Compare claim amounts between fraud and legitimate claims
print(f"\nClaim Amount by Fraud Status:")
print(data.groupby('IsFraud')['ClaimAmount'].describe())

# 3.4: Analyze provider-related features
print("\n" + "-"*80)
print("PROVIDER ANALYSIS")
print("-"*80)
print(f"\nProvider Specialties:")
print(data['ProviderSpecialty'].value_counts().head(10))

# Analyze providers with highest fraud counts
print(f"\nProviders with Fraud History:")
high_fraud_providers = data[data['ProviderFraudCount'] > 0]['ProviderSpecialty'].value_counts()
print(high_fraud_providers.head(10))

# 3.5: Analyze patient demographics
print("\n" + "-"*80)
print("PATIENT DEMOGRAPHICS")
print("-"*80)
print(f"\nGender Distribution:")
print(data['PatientGender'].value_counts())
print(f"\nAge Statistics:")
print(data['PatientAge'].describe())

# 3.6: Analyze insurance types
print("\n" + "-"*80)
print("INSURANCE TYPE ANALYSIS")
print("-"*80)
print(f"\nInsurance Type Distribution:")
print(data['InsuranceType'].value_counts())
print(f"\nFraud Rate by Insurance Type:")
fraud_by_insurance = data.groupby('InsuranceType')['IsFraud'].mean() * 100
print(fraud_by_insurance.sort_values(ascending=False))


# ============================================================================
# SECTION 4: DATA VISUALIZATION
# ============================================================================
# Visualizations help us see patterns that might not be obvious in numbers

print("\n" + "="*80)
print("STEP 3: DATA VISUALIZATION")
print("="*80)
print("\nCreating visualizations... This may take a moment.")

# Create a figure with multiple subplots
# A figure is like a canvas that can hold multiple plots
fig = plt.figure(figsize=(20, 15))

# Visualization 1: Fraud Distribution (Pie Chart)
# Shows the proportion of fraudulent vs legitimate claims
ax1 = plt.subplot(3, 3, 1)
fraud_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                  labels=['Legitimate', 'Fraudulent'],
                  colors=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Fraud vs Legitimate Claims', fontsize=12, fontweight='bold')
plt.ylabel('')

# Visualization 2: Claim Amount Distribution by Fraud Status (Box Plot)
# Box plots show the distribution and outliers in the data
ax2 = plt.subplot(3, 3, 2)
sns.boxplot(x='IsFraud', y='ClaimAmount', data=data)
plt.title('Claim Amount Distribution by Fraud Status', fontsize=12, fontweight='bold')
plt.xlabel('Is Fraud (0=No, 1=Yes)')
plt.ylabel('Claim Amount ($)')
# Add thousands separator to y-axis
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Visualization 3: Top 10 Provider Specialties
# Bar chart showing which specialties have the most claims
ax3 = plt.subplot(3, 3, 3)
data['ProviderSpecialty'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 Provider Specialties', fontsize=12, fontweight='bold')
plt.xlabel('Number of Claims')
plt.ylabel('Specialty')

# Visualization 4: Fraud Rate by Provider Specialty
# Shows which specialties have the highest fraud rates
ax4 = plt.subplot(3, 3, 4)
fraud_by_specialty = data.groupby('ProviderSpecialty')['IsFraud'].mean().sort_values(ascending=False).head(10)
fraud_by_specialty.plot(kind='barh', color='coral')
plt.title('Top 10 Specialties by Fraud Rate', fontsize=12, fontweight='bold')
plt.xlabel('Fraud Rate')
plt.ylabel('Specialty')

# Visualization 5: Patient Age Distribution
# Histogram showing the distribution of patient ages
ax5 = plt.subplot(3, 3, 5)
plt.hist(data['PatientAge'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Patient Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Visualization 6: Insurance Type Distribution
# Bar chart of different insurance types
ax6 = plt.subplot(3, 3, 6)
data['InsuranceType'].value_counts().plot(kind='bar')
plt.title('Insurance Type Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Insurance Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Visualization 7: Fraud Type Distribution
# Shows different categories of fraud
ax7 = plt.subplot(3, 3, 7)
fraud_data = data[data['IsFraud'] == 1]
fraud_data['FraudType'].value_counts().plot(kind='bar', color='red', alpha=0.7)
plt.title('Types of Fraud Detected', fontsize=12, fontweight='bold')
plt.xlabel('Fraud Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Visualization 8: Claim to Typical Ratio Distribution
# This ratio can indicate unusual claim amounts
ax8 = plt.subplot(3, 3, 8)
sns.histplot(data=data, x='ClaimToTypicalRatio', hue='IsFraud', bins=50, kde=True)
plt.title('Claim to Typical Ratio by Fraud Status', fontsize=12, fontweight='bold')
plt.xlabel('Claim to Typical Ratio')
plt.ylabel('Frequency')

# Visualization 9: Number of Procedures Distribution
# Shows how many procedures are typically claimed
ax9 = plt.subplot(3, 3, 9)
sns.countplot(x='NumberOfProcedures', hue='IsFraud', data=data)
plt.title('Number of Procedures by Fraud Status', fontsize=12, fontweight='bold')
plt.xlabel('Number of Procedures')
plt.ylabel('Count')
plt.legend(title='Is Fraud', labels=['Legitimate', 'Fraudulent'])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('/home/claude/eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations created and saved as 'eda_visualizations.png'")
plt.close()

# Create a correlation heatmap
# Correlation shows how features are related to each other
print("\nCreating correlation heatmap...")
plt.figure(figsize=(14, 10))

# Select only numerical columns for correlation
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
# Remove ID columns as they don't provide useful correlations
numerical_cols = [col for col in numerical_cols if 'ID' not in col]

# Calculate correlation matrix
# Correlation values range from -1 to 1
# 1 means perfect positive correlation, -1 means perfect negative correlation, 0 means no correlation
correlation_matrix = data[numerical_cols].corr()

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap created and saved as 'correlation_heatmap.png'")
plt.close()


# ============================================================================
# SECTION 5: DATA PREPROCESSING
# ============================================================================
# We need to prepare our data for machine learning models

print("\n" + "="*80)
print("STEP 4: DATA PREPROCESSING")
print("="*80)

# 5.1: Create a copy of the data
# Always work with a copy to preserve the original data
df = data.copy()
print("\n✓ Created a working copy of the dataset")

# 5.2: Handle datetime features
# Convert the ClaimDate to datetime format to extract useful features
print("\n" + "-"*80)
print("PROCESSING DATE FEATURES")
print("-"*80)
df['ClaimDate'] = pd.to_datetime(df['ClaimDate'])
print("✓ Converted ClaimDate to datetime format")

# We already have ClaimMonth and ClaimDayOfWeek, so we'll use those
# Drop the ClaimDate column as we've extracted what we need
df = df.drop('ClaimDate', axis=1)
print("✓ Extracted temporal features from ClaimDate")

# 5.3: Select features for modeling
# We need to choose which columns to use for training our model
print("\n" + "-"*80)
print("FEATURE SELECTION")
print("-"*80)

# Drop ID columns and target-related columns that would cause data leakage
# Data leakage = using information in training that wouldn't be available in real prediction
columns_to_drop = ['ClaimID', 'ProviderID', 'PatientID', 'FraudType']

# FraudType is dropped because it's only known AFTER we determine fraud
# The model should predict fraud without knowing the fraud type

print(f"\nDropping columns that won't be used for modeling:")
for col in columns_to_drop:
    print(f"  - {col}")

# Separate features (X) and target variable (y)
# X = all the information we use to make predictions
# y = what we're trying to predict (fraud or not)
X = df.drop(columns_to_drop + ['IsFraud'], axis=1)
y = df['IsFraud']

print(f"\n✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")
print(f"✓ Number of features selected: {X.shape[1]}")

# 5.4: Encode categorical variables
# Machine learning models work with numbers, not text
# We need to convert text categories into numbers
print("\n" + "-"*80)
print("ENCODING CATEGORICAL VARIABLES")
print("-"*80)

# Identify categorical columns (columns with text/categories)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode:")
for col in categorical_columns:
    print(f"  - {col}: {X[col].nunique()} unique values")

# Use Label Encoding to convert categories to numbers
# For example: 'Male' -> 0, 'Female' -> 1
label_encoders = {}  # Store encoders in case we need them later
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    
print(f"\n✓ Encoded {len(categorical_columns)} categorical columns")

# Display the processed features
print(f"\nProcessed features:")
print(X.head())

# 5.5: Check class distribution in target variable
print("\n" + "-"*80)
print("CLASS DISTRIBUTION CHECK")
print("-"*80)
print(f"\nTarget variable distribution:")
print(f"  Class 0 (Legitimate): {(y == 0).sum():,} samples ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"  Class 1 (Fraudulent): {(y == 1).sum():,} samples ({(y == 1).sum()/len(y)*100:.2f}%)")
print(f"\n  ⚠ This is an imbalanced dataset!")
print(f"  We'll use SMOTE (Synthetic Minority Over-sampling) to balance it.")


# ============================================================================
# SECTION 6: SPLITTING THE DATA
# ============================================================================
# We split data into training and testing sets to evaluate model performance

print("\n" + "="*80)
print("STEP 5: SPLITTING THE DATA")
print("="*80)

# Split the data: 80% for training, 20% for testing
# Training set: used to teach the model
# Testing set: used to evaluate how well the model learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% of data for testing
    random_state=42,      # For reproducibility
    stratify=y            # Maintain the same fraud ratio in both sets
)

print(f"\n✓ Data split completed!")
print(f"\nTraining Set:")
print(f"  - Features (X_train): {X_train.shape}")
print(f"  - Target (y_train): {y_train.shape}")
print(f"  - Fraud cases: {(y_train == 1).sum():,}")
print(f"  - Legitimate cases: {(y_train == 0).sum():,}")

print(f"\nTesting Set:")
print(f"  - Features (X_test): {X_test.shape}")
print(f"  - Target (y_test): {y_test.shape}")
print(f"  - Fraud cases: {(y_test == 1).sum():,}")
print(f"  - Legitimate cases: {(y_test == 0).sum():,}")


# ============================================================================
# SECTION 7: HANDLING CLASS IMBALANCE WITH SMOTE
# ============================================================================
# SMOTE creates synthetic samples of the minority class (fraudulent claims)

print("\n" + "="*80)
print("STEP 6: HANDLING CLASS IMBALANCE")
print("="*80)

print("\n" + "-"*80)
print("APPLYING SMOTE (Synthetic Minority Over-sampling Technique)")
print("-"*80)

print("\nWhat is SMOTE?")
print("  SMOTE creates synthetic (artificial) examples of the minority class")
print("  by interpolating between existing minority class samples.")
print("  This helps the model learn better from underrepresented fraudulent cases.")

# Before SMOTE
print(f"\nBEFORE SMOTE:")
print(f"  Legitimate claims: {(y_train == 0).sum():,}")
print(f"  Fraudulent claims: {(y_train == 1).sum():,}")
print(f"  Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# After SMOTE
print(f"\nAFTER SMOTE:")
print(f"  Legitimate claims: {(y_train_balanced == 0).sum():,}")
print(f"  Fraudulent claims: {(y_train_balanced == 1).sum():,}")
print(f"  Ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.2f}:1")

print(f"\n✓ Classes are now balanced!")
print(f"✓ Training set size increased from {len(y_train):,} to {len(y_train_balanced):,}")


# ============================================================================
# SECTION 8: FEATURE SCALING
# ============================================================================
# Scaling ensures all features are on the same scale

print("\n" + "="*80)
print("STEP 7: FEATURE SCALING")
print("="*80)

print("\n" + "-"*80)
print("STANDARDIZING FEATURES")
print("-"*80)

print("\nWhat is Feature Scaling?")
print("  Feature scaling transforms all features to have similar ranges.")
print("  This prevents features with large values from dominating the model.")
print("  StandardScaler transforms features to have mean=0 and std=1")

# Create scaler object
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test
# IMPORTANT: We only fit on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled successfully!")
print(f"\nScaled training set shape: {X_train_scaled.shape}")
print(f"Scaled testing set shape: {X_test_scaled.shape}")

# Show example of scaling effect
print(f"\nExample - Before and After Scaling:")
print(f"  Original mean of first feature: {X_train_balanced.iloc[:, 0].mean():.2f}")
print(f"  Scaled mean of first feature: {X_train_scaled[:, 0].mean():.2f}")
print(f"  Original std of first feature: {X_train_balanced.iloc[:, 0].std():.2f}")
print(f"  Scaled std of first feature: {X_train_scaled[:, 0].std():.2f}")


# ============================================================================
# SECTION 9: BUILDING MACHINE LEARNING MODELS
# ============================================================================
# We'll train multiple models and compare their performance

print("\n" + "="*80)
print("STEP 8: BUILDING MACHINE LEARNING MODELS")
print("="*80)

# Dictionary to store models and their predictions
models = {}
predictions = {}
probabilities = {}

# -------------------------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION
# -------------------------------------------------------------------------
print("\n" + "-"*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-"*80)

print("\nWhat is Logistic Regression?")
print("  A simple but powerful algorithm that predicts the probability")
print("  of a claim being fraudulent based on a linear combination of features.")
print("  Best for: Binary classification problems with linear relationships")

print("\nTraining Logistic Regression model...")
# Create and train the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_balanced)

# Make predictions
lr_predictions = lr_model.predict(X_test_scaled)
lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability of fraud

# Store results
models['Logistic Regression'] = lr_model
predictions['Logistic Regression'] = lr_predictions
probabilities['Logistic Regression'] = lr_probabilities

print("✓ Logistic Regression model trained successfully!")

# -------------------------------------------------------------------------
# MODEL 2: RANDOM FOREST CLASSIFIER
# -------------------------------------------------------------------------
print("\n" + "-"*80)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("-"*80)

print("\nWhat is Random Forest?")
print("  An ensemble of decision trees that vote on the final prediction.")
print("  It builds multiple decision trees and combines their predictions.")
print("  Best for: Complex patterns and non-linear relationships")
print("  Advantage: Can handle complex interactions between features")

print("\nTraining Random Forest model...")
# Create and train the model
# n_estimators = number of trees in the forest
# max_depth = maximum depth of each tree (prevents overfitting)
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    random_state=42,
    n_jobs=-1              # Use all CPU cores for faster training
)
rf_model.fit(X_train_scaled, y_train_balanced)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

# Store results
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_predictions
probabilities['Random Forest'] = rf_probabilities

print("✓ Random Forest model trained successfully!")

# -------------------------------------------------------------------------
# MODEL 3: ISOLATION FOREST (ANOMALY DETECTION)
# -------------------------------------------------------------------------
print("\n" + "-"*80)
print("MODEL 3: ISOLATION FOREST (UNSUPERVISED ANOMALY DETECTION)")
print("-"*80)

print("\nWhat is Isolation Forest?")
print("  An unsupervised algorithm designed specifically for anomaly detection.")
print("  It works by isolating anomalies instead of profiling normal points.")
print("  Best for: Detecting unusual patterns without labeled fraud data")
print("  Note: This is an unsupervised method - it doesn't use fraud labels!")

print("\nTraining Isolation Forest model...")
# Train on the original unbalanced training data (without labels)
iso_model = IsolationForest(
    contamination=0.1,     # Expected proportion of outliers (10%)
    random_state=42,
    n_jobs=-1
)

# Isolation Forest is trained without labels (unsupervised)
iso_model.fit(X_train_scaled)

# Make predictions
# Isolation Forest returns 1 for normal, -1 for anomaly
# We convert -1 to 1 (fraud) and 1 to 0 (normal) to match our format
iso_predictions_raw = iso_model.predict(X_test_scaled)
iso_predictions = np.where(iso_predictions_raw == -1, 1, 0)

# Isolation Forest provides anomaly scores instead of probabilities
iso_scores = iso_model.score_samples(X_test_scaled)
# Convert scores to probabilities (higher score = more normal)
iso_probabilities = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# Store results
models['Isolation Forest'] = iso_model
predictions['Isolation Forest'] = iso_predictions
probabilities['Isolation Forest'] = iso_probabilities

print("✓ Isolation Forest model trained successfully!")

print("\n" + "="*80)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*80)


# ============================================================================
# SECTION 10: MODEL EVALUATION
# ============================================================================
# Evaluate how well each model performs

print("\n" + "="*80)
print("STEP 9: MODEL EVALUATION")
print("="*80)

print("\n" + "-"*80)
print("EVALUATION METRICS EXPLANATION")
print("-"*80)
print("\nKey Metrics for Fraud Detection:")
print("  1. Accuracy: Overall correctness (but can be misleading with imbalanced data)")
print("  2. Precision: Of all predicted frauds, how many were actually fraud?")
print("     - High precision = Few false alarms")
print("  3. Recall (Sensitivity): Of all actual frauds, how many did we catch?")
print("     - High recall = We catch most frauds")
print("  4. F1-Score: Harmonic mean of Precision and Recall")
print("     - Balances precision and recall")
print("  5. ROC-AUC Score: Area Under the ROC Curve")
print("     - Measures ability to distinguish between classes")
print("     - Score of 1.0 is perfect, 0.5 is random guessing")

# Create a summary table for all models
evaluation_results = []

for model_name in models.keys():
    print("\n" + "="*80)
    print(f"EVALUATING: {model_name.upper()}")
    print("="*80)
    
    # Get predictions for this model
    y_pred = predictions[model_name]
    y_prob = probabilities[model_name]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC score (only for models with probabilities)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = None
    
    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    # Shows: True Positives, True Negatives, False Positives, False Negatives
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Legit  Fraud")
    print(f"  Actual Legit  {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"         Fraud  {cm[1][0]:6d} {cm[1][1]:6d}")
    
    # Interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nInterpretation:")
    print(f"  ✓ True Positives (TP):  {tp:,} - Correctly identified frauds")
    print(f"  ✓ True Negatives (TN):  {tn:,} - Correctly identified legitimate claims")
    print(f"  ✗ False Positives (FP): {fp:,} - Legitimate claims flagged as fraud")
    print(f"  ✗ False Negatives (FN): {fn:,} - Frauds that were missed")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Legitimate', 'Fraudulent'],
                               digits=4))
    
    # Store results for comparison
    evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc else 'N/A',
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'True Negatives': tn
    })

# Create comparison DataFrame
comparison_df = pd.DataFrame(evaluation_results)

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print("\n")
print(comparison_df.to_string(index=False))

# Determine best model
print("\n" + "-"*80)
print("BEST MODEL RECOMMENDATION")
print("-"*80)

# For fraud detection, we often prioritize recall (catching frauds)
# but also want good precision (avoiding false alarms)
# F1-score balances both
best_f1_idx = comparison_df['F1-Score'].astype(float).idxmax()
best_model_name = comparison_df.loc[best_f1_idx, 'Model']

print(f"\n🏆 RECOMMENDED MODEL: {best_model_name}")
print(f"   Based on highest F1-Score (best balance of precision and recall)")
print(f"\n   Key Metrics:")
print(f"   - F1-Score: {comparison_df.loc[best_f1_idx, 'F1-Score']:.4f}")
print(f"   - Precision: {comparison_df.loc[best_f1_idx, 'Precision']:.4f}")
print(f"   - Recall: {comparison_df.loc[best_f1_idx, 'Recall']:.4f}")

print("\n   Why this matters:")
print(f"   - Will correctly identify {comparison_df.loc[best_f1_idx, 'True Positives']} frauds")
print(f"   - Will miss only {comparison_df.loc[best_f1_idx, 'False Negatives']} frauds")
print(f"   - Will incorrectly flag {comparison_df.loc[best_f1_idx, 'False Positives']} legitimate claims")


# ============================================================================
# SECTION 11: VISUALIZATION OF MODEL PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("STEP 10: VISUALIZING MODEL PERFORMANCE")
print("="*80)

# Create performance comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Model Comparison - Metrics
ax1 = axes[0, 0]
metrics_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Model')
metrics_df.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_xlabel('Model')
ax1.legend(loc='lower right')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Confusion Matrix for Best Model
ax2 = axes[0, 1]
best_cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# Plot 3: ROC Curves
ax3 = axes[1, 0]
for model_name in models.keys():
    if model_name != 'Isolation Forest':  # Skip Isolation Forest for ROC
        y_prob = probabilities[model_name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax3.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

ax3.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=2)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance (for Random Forest)
ax4 = axes[1, 1]
if 'Random Forest' in models:
    # Get feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    feature_importance.plot(x='Feature', y='Importance', kind='barh', ax=ax4, legend=False)
    ax4.set_title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Importance Score')
    ax4.set_ylabel('Feature')

plt.tight_layout()
plt.savefig('/home/claude/model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model performance visualizations saved as 'model_performance_comparison.png'")
plt.close()


# ============================================================================
# SECTION 12: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 11: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if 'Random Forest' in models:
    print("\n" + "-"*80)
    print("ANALYZING FEATURE IMPORTANCE")
    print("-"*80)
    
    # Get feature importances from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print("="*80)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['Feature']:30s} : {row['Importance']:.4f}")
    
    print("\n" + "-"*80)
    print("INTERPRETATION OF TOP FEATURES")
    print("-"*80)
    
    top_feature = feature_importance.iloc[0]
    print(f"\nMost Important Feature: {top_feature['Feature']}")
    print(f"  This feature has the highest impact on fraud detection.")
    print(f"  The model relies on this feature {top_feature['Importance']*100:.2f}% relative to others.")


# ============================================================================
# SECTION 13: BUSINESS INSIGHTS AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 12: BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("="*80)

print("\n" + "-"*80)
print("KEY FINDINGS")
print("-"*80)

# Calculate potential savings
total_frauds = (y_test == 1).sum()
detected_frauds = comparison_df.loc[best_f1_idx, 'True Positives']
missed_frauds = comparison_df.loc[best_f1_idx, 'False Negatives']
avg_fraud_amount = data[data['IsFraud'] == 1]['ClaimAmount'].mean()

print(f"\n1. FRAUD DETECTION PERFORMANCE:")
print(f"   - Total fraudulent claims in test set: {total_frauds:,}")
print(f"   - Successfully detected: {detected_frauds:,} ({detected_frauds/total_frauds*100:.1f}%)")
print(f"   - Missed frauds: {missed_frauds:,} ({missed_frauds/total_frauds*100:.1f}%)")

print(f"\n2. FINANCIAL IMPACT:")
print(f"   - Average fraudulent claim amount: ${avg_fraud_amount:,.2f}")
print(f"   - Potential savings from detected frauds: ${detected_frauds * avg_fraud_amount:,.2f}")
print(f"   - Potential loss from missed frauds: ${missed_frauds * avg_fraud_amount:,.2f}")

print(f"\n3. OPERATIONAL EFFICIENCY:")
false_positives = comparison_df.loc[best_f1_idx, 'False Positives']
print(f"   - False alarms (legitimate claims flagged): {false_positives:,}")
print(f"   - These claims will need manual review")
print(f"   - Precision rate: {comparison_df.loc[best_f1_idx, 'Precision']*100:.1f}%")
print(f"     (Of flagged claims, {comparison_df.loc[best_f1_idx, 'Precision']*100:.1f}% are actually fraudulent)")

print("\n" + "-"*80)
print("RECOMMENDATIONS")
print("-"*80)

print("\n1. IMPLEMENTATION STRATEGY:")
print("   ✓ Deploy the Random Forest model as the primary fraud detection system")
print("   ✓ Use it to automatically flag high-risk claims for review")
print("   ✓ Set up a feedback loop to continuously improve the model")

print("\n2. RISK AREAS TO MONITOR:")
# Get high fraud rate specialties
high_fraud_specialties = data.groupby('ProviderSpecialty')['IsFraud'].agg(['mean', 'sum'])
high_fraud_specialties = high_fraud_specialties[high_fraud_specialties['sum'] > 10].sort_values('mean', ascending=False).head(5)
print("   High-risk provider specialties:")
for spec, row in high_fraud_specialties.iterrows():
    print(f"   - {spec}: {row['mean']*100:.1f}% fraud rate ({int(row['sum'])} cases)")

print("\n3. PROCESS IMPROVEMENTS:")
print("   ✓ Prioritize manual review of high-value claims")
print("   ✓ Implement real-time scoring for new claims")
print("   ✓ Create alerts for claims with unusual patterns")
print("   ✓ Regular model retraining with new fraud cases")

print("\n4. MONITORING METRICS:")
print("   ✓ Track detection rate monthly")
print("   ✓ Monitor false positive rate to avoid claim processing delays")
print("   ✓ Measure financial impact of detected vs missed frauds")
print("   ✓ Review and update model quarterly")


# ============================================================================
# SECTION 14: SAVING THE MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 13: SAVING THE MODEL")
print("="*80)

import pickle

# Save the best performing model
model_filename = '/home/claude/fraud_detection_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(models[best_model_name], file)
print(f"\n✓ Best model ({best_model_name}) saved as 'fraud_detection_model.pkl'")

# Save the scaler
scaler_filename = '/home/claude/scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"✓ Feature scaler saved as 'scaler.pkl'")

# Save the label encoders
encoders_filename = '/home/claude/label_encoders.pkl'
with open(encoders_filename, 'wb') as file:
    pickle.dump(label_encoders, file)
print(f"✓ Label encoders saved as 'label_encoders.pkl'")

# Save the comparison results
comparison_df.to_csv('/home/claude/model_comparison_results.csv', index=False)
print(f"✓ Model comparison results saved as 'model_comparison_results.csv'")


# ============================================================================
# SECTION 15: PROJECT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)

print("\n📊 WHAT WE ACCOMPLISHED:")
print("-"*80)
print("1. ✓ Loaded and explored 100,000 healthcare insurance claims")
print("2. ✓ Performed comprehensive exploratory data analysis")
print("3. ✓ Visualized key patterns and relationships in the data")
print("4. ✓ Preprocessed data (encoding, scaling, balancing)")
print("5. ✓ Handled class imbalance using SMOTE technique")
print("6. ✓ Built and trained 3 different machine learning models:")
print("     - Logistic Regression")
print("     - Random Forest Classifier")
print("     - Isolation Forest (Anomaly Detection)")
print("7. ✓ Evaluated and compared model performance")
print("8. ✓ Identified the best performing model")
print("9. ✓ Analyzed feature importance")
print("10. ✓ Generated business insights and recommendations")
print("11. ✓ Saved models and results for future use")

print("\n🎯 KEY DELIVERABLES:")
print("-"*80)
print("1. fraud_detection_model.pkl - Trained machine learning model")
print("2. scaler.pkl - Feature scaler for preprocessing")
print("3. label_encoders.pkl - Encoders for categorical variables")
print("4. eda_visualizations.png - Exploratory data analysis charts")
print("5. correlation_heatmap.png - Feature correlation analysis")
print("6. model_performance_comparison.png - Model evaluation charts")
print("7. model_comparison_results.csv - Detailed metrics comparison")

print("\n💡 KEY LEARNINGS:")
print("-"*80)
print("• Healthcare fraud detection is a challenging imbalanced classification problem")
print("• Feature engineering and proper preprocessing are crucial for model performance")
print("• Ensemble methods (Random Forest) often outperform simple models")
print("• Balancing precision and recall is important in fraud detection")
print("• Regular model updates are necessary to catch evolving fraud patterns")

print("\n🚀 NEXT STEPS:")
print("-"*80)
print("1. Deploy the model in a production environment")
print("2. Set up automated retraining pipeline")
print("3. Create a real-time scoring API")
print("4. Build a dashboard for monitoring fraud detection")
print("5. Implement A/B testing for model improvements")
print("6. Gather feedback from fraud investigation team")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*80)
print("\nThank you for using this fraud detection system!")
print("For questions or improvements, please refer to the documentation.\n")
