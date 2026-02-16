# Crop Recommendation System using Ensemble Learning

## Overview

This project implements a **Crop Recommendation System** using Machine Learning techniques.  

The model predicts the most suitable crop based on environmental and soil parameters using an **ensemble approach (Soft Voting Classifier)**.

The system combines:

- Random Forest  
- Support Vector Machine (SVM)  
- Gradient Boosting  

The goal is to improve prediction accuracy by leveraging multiple models.

---

## Dataset

The project uses:

```
crop_recommendation.csv
```

The dataset contains agricultural and environmental parameters such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall
- Label (Crop type)

The `label` column represents the crop to be predicted.

---

## Model Pipeline

### 1. Data Loading & Preprocessing

- Load dataset using Pandas  
- Check for missing values  
- Separate features (X) and target (y)  
- Split into training and test sets (80% / 20%)  
- Apply StandardScaler for feature scaling  

---

### 2. Models Used

- **RandomForestClassifier**
- **Support Vector Machine (Linear Kernel)**
- **GradientBoostingClassifier**

---

### 3. Ensemble Method

A **VotingClassifier (Soft Voting)** is used to combine predictions.

Soft voting averages predicted probabilities from:

- Random Forest
- SVM
- Gradient Boosting

This improves stability and accuracy compared to single models.

---

## Evaluation Metrics

- Accuracy Score
- Classification Report
  - Precision
  - Recall
  - F1-Score
  - Support

---

## Project Structure

```
/crop-recommendation-system
├── crop_recommendation.csv
├── crop_recommendation.py
└── README.md
```

---

## Execution Instructions

### Prerequisites

- Python 3.8+
- Required libraries installed

---

### Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

---

### Step 2: Ensure Dataset is Present

Make sure the file:

```
crop_recommendation.csv
```

is in the same directory as:

```
crop_recommendation.py
```

---

### Step 3: Run the Script

```bash
python crop_recommendation.py
```

---

## Output

The program will display:

- First few rows of dataset
- Dataset summary statistics
- Missing values check
- Model Accuracy
- Full Classification Report

Example:

```
Accuracy: 98.75%

Classification Report:
              precision    recall    f1-score   support
```

---

## Machine Learning Concepts Used

- Train-Test Split
- Feature Scaling (StandardScaler)
- Ensemble Learning
- Soft Voting
- Classification Metrics

---

## Conclusion

This project demonstrates the use of **ensemble machine learning techniques** to build an accurate crop recommendation system based on soil and environmental features.

It highlights practical application of:

- Supervised Learning
- Model Combination Techniques
- Performance Evaluation in Classification Problems
