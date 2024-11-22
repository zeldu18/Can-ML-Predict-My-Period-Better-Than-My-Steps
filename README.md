# Can-ML-Predict-My-Period-Better-Than-My-Steps

The quick answer is **yes and no** :).  
This project explores the potential of machine learning models to predict menstrual cycle patterns and daily step counts based on iOS Health Indexes and self-tracked period data. Using advanced machine learning techniques, we investigate two key questions:  
- Can ML accurately predict if I’m on my period based on my health data?  
- Can ML effectively predict the number of steps I take daily?  

---

## Table of Contents
- [Project Motivation](#project-motivation)
- [Data Overview](#data-overview)
- [Methods and Models](#methods-and-models)
- [Results](#results)
- [Key Insights](#key-insights)
- [How to Use This Repository](#how-to-use-this-repository)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

## Project Motivation

The curiosity behind this project stems from years of tracking health and period data. With access to detailed iOS Health Indexes (e.g., step count, walking speed, energy burned) and period tracking, this project aims to answer:  
- **How well can machine learning predict health patterns?**  
- **Is it possible for ML to outperform simpler baseline methods?**  

This study also dives into the impact of **class imbalance**, **feature engineering**, and **hyperparameter tuning** on prediction performance.

---

## Data Overview

The dataset combines 5.5 years of personal health data collected from:  

### iOS Health App:
- `steps`, `distance_walked`, `walking_speed`, `energy_burned`, and categorical metadata (e.g., `timeZone`, `day_of_week`).

### Flo App:
- Period tracking data (`is_on_period`).

### Preprocessing Steps:
- Removed perfectly correlated features (e.g., `distance_walked`).
- One-hot encoded categorical features (e.g., `timeZone`).
- Addressed class imbalance using **class weighting** or **oversampling**.
- Standardized numerical features for certain models (e.g., Neural Networks).

---

## Methods and Models

### Classification: Predicting Period Days  
**Models used:**  
- **Decision Trees**:  
  - Basic Decision Tree  
  - Class-Weighted Decision Tree  
  - Hyperparameter-Tuned Decision Tree  

- **Neural Networks (MLP)**:  
  - Basic MLP  
  - Oversampled MLP  
  - Hyperparameter-Tuned MLP  

**Error metrics evaluated:**  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

Baseline comparison: A **constant classifier** predicting the majority class (No Period).  

---

### Regression: Predicting Step Counts  
**Models used:**  
- **Random Forest Regression**:  
  - Basic Random Forest  
  - Hyperparameter-Tuned Random Forest  

- **Neural Networks (MLP Regressor)**:  
  - Basic MLP Regressor  
  - Hyperparameter-Tuned MLP Regressor  

**Error metrics evaluated:**  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- R-squared (R²)  

Residual and feature importance plots provided further insights.

---

## Results

### Classification Results:
- **Best F1 Score**: 0.31 using an **Optimized Decision Tree**.  
- The **Baseline Model** (predicting "No Period") achieved an **accuracy of 80%**, but with an **F1 Score of 0**.  
- Neural Networks struggled due to the dataset's limited size and complexity.

### Regression Results:
- **Random Forest Regression**:  
  - Basic: R² = 0.395  
  - Optimized: R² = 0.453  

- **Neural Network Regression**:  
  - Performed worse overall with R² below 0.2, indicating difficulties in capturing step count patterns.

---

## Key Insights

- **Period Prediction is Challenging**:  
  - The data lacks significant variation between period and non-period days.  
  - Even the best models struggled to outperform the baseline significantly.  

- **Step Prediction is Feasible**:  
  - Random Forests outperformed Neural Networks in predicting step counts.  
  - Feature importance analysis revealed that `energy_burned` and `walking_speed` were the strongest predictors.  

- **Class Imbalance Matters**:  
  - Addressing class imbalance via **class weighting** and **oversampling** improved results slightly but not significantly.  

- **Hyperparameter Tuning**:  
  - Grid Search and Randomized Search were effective but had diminishing returns for models like Neural Networks due to convergence issues.  

---

## How to Use This Repository

### Prerequisites:
- Python 3.8+  
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.  

### Clone the Repository:
```bash
git clone https://github.com/yourusername/Can-ML-Predict-My-Period-Better-Than-My-Steps.git
cd Can-ML-Predict-My-Period-Better-Than-My-Steps
