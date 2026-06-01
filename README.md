# 🩺 The Prescription Predictor: Logistic Regression Dosage Classifier

An interactive Supervised Learning simulation designed to teach **Logistic Regression** and the **Sigmoid Activation Function** from scratch. You play as a Clinical Pharmacy Data Scientist calibrating a real-time health-tech monitoring system, transforming raw blood serum concentration vectors into reliable binary toxicity probabilities to intercept critical hospital overdoses.

## 🎓 Learning Objectives

This project focuses on teaching:
* **Logistic Regression:** A foundational classification algorithm that maps independent continuous variables to binary probability profiles.
* **The Sigmoid Function:** The classic mathematical S-curve ($\frac{1}{1 + e^{-z}}$) used to compress infinite linear arrays cleanly into a bound $0.0$ to $1.0$ probability landscape.
* **Linear Log-Odds ($z$):** Understanding how adjusting structural feature weights ($w$) and baseline biases ($b$) shifts the mathematical inflection point of a classification system.
* **Decision Thresholding:** Setting explicit cut-off bounds (e.g., $50\%$) to convert continuous risk percentages into immediate medical override actions.

---

## ✨ Features

* **Intensive Care Clinical Narrative:** Contextualizes probability mapping and feature modeling within an emergency critical-care medical workspace.
* **Algorithmic Transparency Engine:** Details and prints the linear combination calculations and final exponential Sigmoid activation results step-by-step.
* **Live Dynamic Parameter Calibration:** Allows users to manually tweak the regression slope weights and intercepts to instantly observe how parameters influence probability curves.
* **Zero Module Overheads:** Built purely inside native Python, mapping transcendental algebraic functions without importing external numerical arrays or data science packages.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/prescription-predictor-logistic.git](https://github.com/YOUR_USERNAME/prescription-predictor-logistic.git)
    cd prescription-predictor-logistic
    ```
2.  **Save the Code:** Save the provided script as `dosage_regressor.py`.
3.  **Run the Script:**
    ```bash
    python dosage_regressor.py
    ```

### 3. Gameplay Instructions
1.  **Examine Training Baselines:** Review the continuous historical lab concentrations (mg/dL) alongside their binary labels ($0 = \text{Safe}$, $1 = \text{Overdose}$).
2.  **Calibrate the Parameters:** Input values for the structural model Weight and Bias to position your logistic decision boundary.
3.  **Audit the Emergency Queue:** Monitor the script as a high-stakes borderline prescription concentration reading ($32.0\text{ mg/dL}$) hits the network.
4.  **Evaluate Probability Metrics:** Check the activated percentage score to confirm if your hyperplanes successfully isolated the clinical hazard.

---

## 🧠 Code Structure Highlights

### Linear Combination Log-Odds Matrix
The script calculates the intermediate spatial score ($z$) by computing the continuous product of raw feature inputs and matching parameters.

```python
# Linear Step: z = (Feature * Weight) + Bias
z = (test_concentration * weight) + bias

