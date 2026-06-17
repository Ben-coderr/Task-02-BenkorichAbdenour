# Data Science — Project 2
### Industrial Training Kit
**Batch: 2026 | Powered by DecodeLabs**

---

## WELCOME TO THE TEAM! 🚀

Step into the role of a Data Scientist at DecodeLabs. Project 2 is your vital bridge: Supervised Learning (Fraud Detection Pipeline).

This track isn't about "just predicting outcomes" — it's about **Algorithmic Precision**.

Now that you have mastered data wrangling, you must master the art of handling highly imbalanced datasets using SMOTE, training robust classification models like Random Forest, and completely discarding misleading 'Accuracy' metrics in favor of strict Precision, Recall, and ROC-AUC evaluation. By completing this milestone, you are proving you can identify hidden anomalies and protect financial ecosystems through pure machine learning logic.

**Let's build a pipeline that detects fraud with absolute clarity.**

---

## Project 2: Supervised Learning (Fraud Detection Pipeline)

**Goal:** Build and tune a classification model to identify fraudulent transactions in a highly imbalanced dataset.

### Key Requirements:
- Implement **SMOTE** (Synthetic Minority Over-sampling) to handle class imbalance.
- Train multiple algorithms (**Logistic Regression**, **Random Forest**) using Scikit-Learn.
- Discard **"Accuracy"** and evaluate the model using strict **Precision**, **Recall**, and **ROC-AUC** metrics.

### Key Skills:
Classification algorithms, Scikit-Learn pipelines, imbalanced data handling, hyperparameter tuning.

---

## The Leak-Free Pipeline
### Architecting Robust Fraud Detection Systems with SMOTE and Imblearn
*Data Science Intern Masterclass*

---

## The Reality of Financial Datasets

| Metric | Value |
|---|---|
| **Total Sample Pool** | **284,807 Transactions** |
| **Legitimate Rate** | **99.83%** |
| **Fraudulent Rate** | **0.17%** |

Machine learning models are inherently lazy. When legitimate transactions represent over 99.83% of total volume, global prediction error minimization drives models to simply predict "Legitimate" every time.

---

## Trap #1: The Illusion of Accuracy

A model that classifies **every** transaction as legitimate achieves **99.83% Accuracy** — but catches **Zero Fraud**.

### Confusion Matrix Breakdown:

| | Predicted Legitimate | Predicted Fraudulent |
|---|---|---|
| **Actual Legitimate** | True Negative (TN) | False Positive (FP) |
| **Actual Fraudulent** | **False Negative (FN)** ⚠️ | True Positive (TP) |

> **Zero Fraud Caught. 99.83% Accuracy.**

In enterprise payment infrastructures, a model that classifies every transaction as legitimate achieves near-perfect accuracy while resulting in **catastrophic financial loss**.

---

## The True Compass

### 1. Precision
**Formula:** `TP / (TP + FP)`

When we flag fraud, are we right? Minimizes false declines and customer frustration.

### 2. Recall
**Formula:** `TP / (TP + FN)`

Did we catch all the fraud? Missing a fraudulent event results in direct financial loss.

### 3. ROC-AUC
**Target: 0.85+**

The model's overall capability to separate the legitimate distribution from the fraudulent distribution.

---

## Rebalancing the Scales

### Three Approaches Compared:

| Approach | Name | Description |
|---|---|---|
| **Undersampling** | The Loss | Destroys valuable baseline data. |
| **Oversampling** | The Echo | Mere duplication. Leads to severe model overfitting. |
| **SMOTE** | The Synthesis | Synthetic Minority Over-sampling Technique. **SMOTE doesn't clone; it creates.** |

---

## How SMOTE Interpolates

**Formula:**
```
x_new = x_i + λ × (x_nn − x_i)
```

Where **λ ~ Uniform(0, 1)**

A random interpolation weight is drawn, populating sparse regions of the minority feature space to help classifiers learn a robust decision boundary.

---

## Trap #2: The Data Leakage Catastrophe

### ❌ WRONG Order (Causes Leakage):
```
Entire Imbalanced Dataset → SMOTE (Data generation) → ✗ 80/20 Train/Test Split
```

**Data leakage occurs when information that would not be available at prediction time is used when building the model. Applying SMOTE before splitting means you are testing the model on synthetic data that already knows the training answers.**

---

## The API Imperative: Sklearn vs. Imblearn

### ❌ Fails in Production: `sklearn.pipeline.Pipeline`
Standard pipeline steps expect a transform method that only modifies the feature matrix (X). Resampling is ignored or crashes.

```
Transform(X) → ✗ Split → Train
```

### ✅ Production Standard: `imblearn.pipeline.Pipeline`
Natively supports resampling. Modifies both the feature matrix (X) and the target vector (y) strictly on the training fold via the `fit_resample` interface.

```
Split → SMOTE(X, y) → Train → Predict
```

---

## The Golden Rule of Validation

**5-Fold Cross Validation Flow:**

```
80% Training Fold → SMOTE Chamber → Model Training
                                          ↓
20% Validation Fold ←←←← Model Prediction
```

> **Never expose your validation fold to scaling or resampling. Your blind exam must reflect the extreme imbalance of the real world.**

---

## Choosing the Prediction Engine

| Model | Speed & Interpretability | Scaling Sensitivity | Decision Boundary |
|---|---|---|---|
| **Logistic Regression** | High (Coefficient transparency) | **Fatal without Scaler.** Regularization penalties are highly distorted by massive transaction amount variances. | Linear |
| **Random Forest** | Slower (Ensemble logic), moderate interpretability | **Immune.** Splits are based purely on ordinal feature partitions, invariant to scale. | Highly complex, non-linear |

---

## The Engine Room: GridSearchCV

GridSearchCV safely applies SMOTE inside every fold calculation for every parameter combination, ensuring **zero leakage during hyperparameter tuning**.

### Hyperparameter Grid Example:

**The Preprocessor (SMOTE):**
- `smote__k_neighbors = [3]`
- `smote__k_neighbors = [5]`
- `smote__k_neighbors = [7]`

**The Engine (Classifier):**
- LR: `classifier__C = [0.01]`
- `classifier__C = [0.1]`
- `classifier__C = [1.0]`
- RF: `classifier__max_depth = [10]`
- RF: `classifier__max_depth = [20]`
- RF: `classifier__max_depth = [None]`

---

## Constructing the Linear Pipeline

```python
steps = [('scaler', StandardScaler()),
         ('smote', SMOTE()),
         ('classifier', LogisticRegression())]
```

### Pipeline Flow:
1. **Standardization** (μ=0, σ=1)
2. **Synthetic Balancing**
3. **Gradient Descent Scoring**

> ⚠️ **StandardScaler must live inside the pipeline. Scaling globally before splitting leaks mean and variance statistics from the test partition.**

---

## Constructing the Ensemble Pipeline

```python
steps = [('smote', SMOTE()),
         ('classifier', RandomForestClassifier())]
```

### Pipeline Flow:
1. *(No StandardScaler required)*
2. **Synthetic Balancing**
3. **Ensemble Tree Scoring**

> ⚠️ No **StandardScaler** required. Tree-based models partition feature space ordinally, rendering scale transformations mathematically irrelevant.

---

## Architectural Synthesis — The Imblearn Pipeline Loop

```
Raw Imbalanced Data        Stratified Split
(99.83% Cyan /      →     (Ensuring exact    →  [  Scaler   →   SMOTE    →    Model    ]  →  GridSearchCV  →  Final Evaluation
 0.17% Coral)              class ratios)          (if LR)     Balancing    Gradient           Tuning          (ROC-AUC &
                                                                           Descent/          (Multi-fold      Confusion Matrix
                                                                           Ensemble Tree      iteration)       on untouched
                                                                           Scoring)                            test data)
```

> **Scalable, tunable, and mathematically secure against data leakage.**

---

## The Zero-Leakage Protocol

- ✅ **Ditch Accuracy.** Optimize models using Recall, F1, and ROC-AUC.
- ✅ Use **SMOTE** to interpolate and generate, **never** just duplicate.
- ✅ **NEVER** apply SMOTE or Scalers prior to the Train/Test Split.
- ✅ **ALWAYS** utilize `imblearn.pipeline.Pipeline` to safely isolate resampling within Cross-Validation.
- ✅ Tune preprocessing and model hyperparameters holistically inside GridSearchCV.

> **Deploy with precision.**

---

## CONCLUSION

The absolute best way to master Supervised Learning is through rigorous model evaluation, not just theory. Don't just aim to complete these projects; take them one by one, experiment with unique solutions — like tuning the hyperparameters of your Logistic Regression or comparing different synthetic sampling strategies — and treat every "false positive" as a valuable learning opportunity. As you build these milestones at DecodeLabs, you are creating a real-world analytical portfolio that showcases your machine learning proficiency to future tech firms or financial institutions. Your journey to becoming a professional Data Scientist accelerates right here, right now, with the very first classification algorithm you train today.

---

## THANK YOU

📞 +91 9236011887
✉ decodelabs.tech@gmail.com
🌎 www.decodelabs.tech
📍 GREATER LUCKNOW, INDIA