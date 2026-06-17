#  Data Science Project 2: Fraud Detection Pipeline
**DecodeLabs Internship | Batch 2026**

##  Overview
A robust classification pipeline designed to catch credit card fraud in highly imbalanced datasets (~0.17% fraud rate). This project explicitly demonstrates the dangers of the "Accuracy Trap" and enforces a **Zero-Leakage Architecture**.

##  Architecture & Features
* **Zero-Leakage Protocol:** `imblearn.pipeline.Pipeline` is strictly used to isolate `StandardScaler` and `SMOTE` inside the cross-validation loops, preventing future test data from leaking into the training scaler/synthesizer.
* **Imbalanced Learning:** Employs Synthetic Minority Over-sampling Technique (SMOTE) to mathematically interpolate fraudulent examples without blindly duplicating data.
* **Holistic Tuning:** Uses `GridSearchCV` combined with `StratifiedKFold` to simultaneously tune both the synthesizer (`k_neighbors`) and the classifier (`C` or `max_depth`).
* **Algorithmic Duality:** Evaluates two structurally distinct models:
  * *Logistic Regression:* Highly sensitive to scale, showcasing the need for `StandardScaler`.
  * *Random Forest:* A scale-invariant tree ensemble demonstrating robustness against non-linear distributions.
* **Advanced Metrics:** Abandons Accuracy in favor of ROC-AUC, Precision, Recall, and F1-Score to truthfully measure fraud detection capability.

##  Repository Structure
* `project2_solution.ipynb`: Professional Jupyter Notebook containing the full training pipeline, learning curves, and evaluation heatmaps.
* `project2_solution.py`: Command-line interface version of the pipeline.
* `model_comparison_results.csv`: Tabular output proving model dominance across various metrics.
* `plots/`: Directory containing dynamically generated ROC curves, Precision-Recall curves, and Confusion Matrices.

##  Getting Started
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib jupyter

# 2. Run the Notebook
jupyter notebook project2_solution.ipynb

# 3. Or run the CLI script
python project2_solution.py
```

##  Key Takeaway
By combining SMOTE with strict cross-validation containment, the model dynamically learns to flag the 0.17% anomalous fraud class without compromising the mathematical integrity of the validation set.

---
*Authored with strict adherence to DecodeLabs architectural guidelines.*
