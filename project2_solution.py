"""
================================================================================
  PROJECT 2 — Supervised Learning: Fraud Detection Pipeline
  DecodeLabs Internship | Batch 2026
  Author: Benkorich Abdenour
================================================================================

  Goal:
    Build and tune a classification model to identify fraudulent transactions
    in a highly imbalanced dataset.

  Key Requirements (from PDF):
    1. Implement SMOTE (Synthetic Minority Over-sampling) to handle class imbalance
    2. Train multiple algorithms (Logistic Regression, Random Forest) via Scikit-Learn
    3. Discard "Accuracy" — evaluate using strict Precision, Recall, and ROC-AUC
    4. Use imblearn pipelines and hyperparameter tuning (GridSearchCV)

  Architectural Constraints (Zero-Leakage Protocol):
      NEVER apply SMOTE or Scalers prior to the Train/Test Split
      ALWAYS use imblearn.pipeline.Pipeline to isolate resampling in CV
      LR pipeline: StandardScaler → SMOTE → LogisticRegression
      RF pipeline: SMOTE → RandomForestClassifier  (no scaler needed)
      Tune SMOTE k_neighbors alongside classifier hyperparameters
      Optimize via ROC-AUC, not Accuracy

  Dataset:
    Synthetic credit-card-style fraud dataset (50,000 transactions, ~0.17% fraud)
    modeled after the Kaggle Credit Card Fraud Detection dataset structure.

================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Leak-free pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Output directory for saved plots
try:
    PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
except NameError:
    PLOT_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 80)
print("  PROJECT 2 — Fraud Detection Pipeline (Zero-Leakage Architecture)")
print("  DecodeLabs Internship | Batch 2026")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA GENERATION & LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 1: Data Generation & Loading")
print("─" * 80)

try:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditcard_fraud.csv")
except NameError:
    DATA_PATH = os.path.join(os.getcwd(), "creditcard_fraud.csv")


def generate_synthetic_fraud_data(n_samples=50000, fraud_ratio=0.0017, seed=42):
    """
    Generate a synthetic credit card fraud dataset with realistic properties.

    The dataset mimics the Kaggle Credit Card Fraud Detection dataset:
      - 28 PCA-transformed features (V1–V28)
      - Time (seconds elapsed from first transaction)
      - Amount (transaction amount in USD)
      - Class (0 = Legitimate, 1 = Fraud)

    Real-world stats the document cites:
      Total Sample Pool:  284,807 transactions
      Legitimate Rate:    99.83%
      Fraudulent Rate:    0.17%

    We use 50,000 samples at the same 0.17% ratio for tractable computation.
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    print(f"  Generating {n_samples:,} synthetic transactions...")
    print(f"    → Legitimate: {n_legit:,} ({(1 - fraud_ratio) * 100:.2f}%)")
    print(f"    → Fraudulent: {n_fraud:,} ({fraud_ratio * 100:.2f}%)")
    print(f"    → Imbalance Ratio: 1:{n_legit // n_fraud}")

    # -- Legitimate transactions --
    V_legit = np.random.randn(n_legit, 28)
    V_legit[:, 0] += 0.5    # V1 tends positive for legit
    V_legit[:, 3] += 0.3    # V4 tends positive for legit
    V_legit[:, 13] -= 0.2   # V14 tends slightly negative for legit

    time_legit = np.sort(np.random.uniform(0, 172800, n_legit))  # ~48 hours
    amount_legit = np.abs(np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit))

    # -- Fraudulent transactions --
    V_fraud = np.random.randn(n_fraud, 28) * 1.5  # higher variance
    V_fraud[:, 0] -= 2.0    # V1 shifts strongly negative
    V_fraud[:, 2] += 1.5    # V3 shifts positive
    V_fraud[:, 3] -= 1.8    # V4 shifts negative
    V_fraud[:, 9] -= 1.2    # V10 shifts negative
    V_fraud[:, 11] += 1.8   # V12 shifts positive
    V_fraud[:, 13] += 2.5   # V14 shifts strongly positive
    V_fraud[:, 16] -= 1.5   # V17 shifts negative

    time_fraud = np.sort(np.random.uniform(0, 172800, n_fraud))
    amount_fraud = np.abs(np.random.lognormal(mean=4.5, sigma=1.8, size=n_fraud))

    # -- Assemble DataFrame --
    V_cols = [f"V{i}" for i in range(1, 29)]

    df_legit = pd.DataFrame(V_legit, columns=V_cols)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(V_fraud, columns=V_cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    cols = ["Time"] + V_cols + ["Amount", "Class"]
    return df[cols]


if os.path.exists(DATA_PATH):
    print("  Loading existing dataset from disk...")
    df = pd.read_csv(DATA_PATH)
else:
    df = generate_synthetic_fraud_data(n_samples=50000, fraud_ratio=0.0017, seed=42)
    df.to_csv(DATA_PATH, index=False)
    print(f"  ✓ Dataset saved to: {DATA_PATH}")

print(f"\n  Dataset shape: {df.shape}")
print(f"  Features: {df.shape[1] - 1} | Target: 'Class'")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 2: Exploratory Data Analysis (EDA)")
print("─" * 80)

# 2.1 — Basic statistics
print("\n  ▸ First 5 rows:")
print(df.head().to_string(max_cols=10))
print(f"\n  ▸ Dataset Summary:")
print(f"    Rows:           {len(df):,}")
print(f"    Columns:        {df.shape[1]}")
print(f"    Missing values: {df.isnull().sum().sum()}")
print(f"    Duplicate rows: {df.duplicated().sum()}")

# 2.2 — Class distribution (The Reality of Financial Datasets)
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print(f"\n  ▸ The Reality of Financial Datasets:")
print(f"    ┌──────────────────────┬──────────────────┐")
print(f"    │ Metric               │ Value            │")
print(f"    ├──────────────────────┼──────────────────┤")
print(f"    │ Total Transactions   │ {len(df):>16,} │")
print(f"    │ Legitimate Rate      │ {class_pct[0]:>15.2f}% │")
print(f"    │ Fraudulent Rate      │ {class_pct[1]:>15.2f}% │")
print(f"    │ Imbalance Ratio      │ {'1:' + str(class_counts[0] // class_counts[1]):>16s} │")
print(f"    └──────────────────────┴──────────────────┘")

# 2.3 — Plot: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#2ecc71", "#e74c3c"]

bars = axes[0].bar(["Legitimate\n(Class 0)", "Fraudulent\n(Class 1)"],
                   class_counts.values, color=colors, edgecolor="black", linewidth=0.8)
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
axes[0].set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Number of Transactions")

axes[1].pie(class_counts.values, labels=["Legitimate", "Fraudulent"],
            autopct="%1.3f%%", colors=colors, startangle=90,
            explode=(0, 0.15), shadow=True,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[1].set_title("Class Distribution (Percentage)", fontsize=14, fontweight="bold")

plt.suptitle("Extreme Class Imbalance in Fraud Detection",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_class_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ Plot saved: plots/01_class_distribution.png")

# 2.4 — Transaction Amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (label, color) in enumerate(zip(["Legitimate", "Fraudulent"], colors)):
    subset = df[df["Class"] == i]["Amount"]
    axes[i].hist(subset, bins=60, color=color, edgecolor="black", alpha=0.85)
    axes[i].set_title(f"{label} — Amount Distribution",
                      fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Amount ($)")
    axes[i].set_ylabel("Frequency")
    axes[i].axvline(subset.median(), color="black", linestyle="--", linewidth=1.5,
                    label=f"Median: ${subset.median():.2f}")
    axes[i].legend(fontsize=10)
plt.suptitle("Transaction Amount Distribution by Class",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_amount_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/02_amount_distribution.png")

# 2.5 — Correlation heatmap
print("\n  ▸ Computing feature correlations with target...")
correlations = (df.corr(numeric_only=True)["Class"]
                .drop("Class").abs().sort_values(ascending=False))
top_features = correlations.head(10).index.tolist()
print(f"    Top 10 correlated features: {top_features}")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[top_features + ["Class"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title("Correlation Heatmap — Top 10 Features + Target",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/03_correlation_heatmap.png")

# 2.6 — Statistical summary
print("\n  ▸ Statistical Summary (Amount):")
for cls, label in zip([0, 1], ["Legitimate", "Fraudulent"]):
    subset = df[df["Class"] == cls]["Amount"]
    print(f"    {label}:")
    print(f"      Mean:   ${subset.mean():.2f}")
    print(f"      Median: ${subset.median():.2f}")
    print(f"      Std:    ${subset.std():.2f}")
    print(f"      Max:    ${subset.max():.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: STRATIFIED TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 3: Stratified Train/Test Split")
print("─" * 80)

print("""
  ▸ ZERO-LEAKAGE PROTOCOL:
    The document explicitly states:
      ❌ WRONG:   Entire Dataset → SMOTE → Train/Test Split  (DATA LEAKAGE!)
       CORRECT: Entire Dataset → Stratified Split → SMOTE only on Train fold

    We split FIRST, then all preprocessing (scaling, SMOTE) happens INSIDE
    the imblearn pipeline — applied only to the training fold during CV.
""")

X = df.drop("Class", axis=1)
y = df["Class"]

# 80/20 stratified split — preserves exact class ratios in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ▸ Features (X): {X.shape}")
print(f"  ▸ Target   (y): {y.shape}")
print(f"\n  ▸ Train set: {X_train.shape[0]:,} samples")
print(f"    → Class 0: {(y_train == 0).sum():>6,}  |  Class 1: {(y_train == 1).sum():,}")
print(f"  ▸ Test set:  {X_test.shape[0]:,} samples")
print(f"    → Class 0: {(y_test == 0).sum():>6,}  |  Class 1: {(y_test == 1).sum():,}")
print(f"\n  ⚠  Test set remains UNTOUCHED — no SMOTE, no scaling applied to it")
print(f"     until prediction time (handled automatically by the pipeline).")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAP #1 — THE ILLUSION OF ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 4: Trap #1 — The Illusion of Accuracy")
print("─" * 80)

print("""
  ▸ A model that classifies EVERY transaction as "Legitimate" achieves
    99.83% Accuracy — but catches ZERO fraud.

    Confusion Matrix of a Naive "All-Legitimate" Classifier:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │ Predicted Legit  │ Predicted Fraud  │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Actual Legit    │    TN (all)      │    FP = 0        │
    │ Actual Fraud    │    FN (all) ⚠️   │    TP = 0        │
    └─────────────────┴──────────────────┴──────────────────┘
    → Zero Fraud Caught. 99.83% Accuracy. Catastrophic financial loss.
""")

# Demonstrate the accuracy trap
y_pred_naive = np.zeros_like(y_test)  # Predict everything as Class 0
naive_acc = accuracy_score(y_test, y_pred_naive)
naive_prec = precision_score(y_test, y_pred_naive, zero_division=0)
naive_rec = recall_score(y_test, y_pred_naive)
naive_f1 = f1_score(y_test, y_pred_naive)

print(f"  ▸ Naive 'All-Legitimate' Classifier Results:")
print(f"    Accuracy:  {naive_acc:.4f}  ({naive_acc * 100:.2f}%)  ← LOOKS PERFECT!")
print(f"    Precision: {naive_prec:.4f}  ← Cannot measure (zero predictions)")
print(f"    Recall:    {naive_rec:.4f}  ← CATCHES ZERO FRAUD")
print(f"    F1-Score:  {naive_f1:.4f}  ← Complete failure")
print(f"\n  ⚠  This is why we DISCARD accuracy and use Precision, Recall, ROC-AUC.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THE TRUE COMPASS — METRICS EXPLAINED
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 5: The True Compass — Evaluation Metrics")
print("─" * 80)

print("""
  ▸ 1. PRECISION = TP / (TP + FP)
    "When we flag fraud, are we right?"
    Minimizes false declines and customer frustration.

  ▸ 2. RECALL = TP / (TP + FN)
    "Did we catch all the fraud?"
    Missing a fraudulent event = direct financial loss.

  ▸ 3. ROC-AUC  (Target: 0.85+)
    The model's overall capability to separate the legitimate distribution
    from the fraudulent distribution.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SMOTE — REBALANCING THE SCALES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 6: SMOTE — Rebalancing the Scales")
print("─" * 80)

print("""
  ▸ Three Approaches Compared:
    ┌──────────────────┬────────────────────────────────────────────────────┐
    │ Approach         │ Description                                        │
    ├──────────────────┼────────────────────────────────────────────────────┤
    │ Undersampling    │ Destroys valuable baseline data. (THE LOSS)        │
    │ Oversampling     │ Mere duplication → severe overfitting. (THE ECHO)  │
    │ SMOTE            │ Synthetic interpolation. Creates, doesn't clone.   │
    │                  │ (THE SYNTHESIS)                                   │
    └──────────────────┴────────────────────────────────────────────────────┘

  ▸ How SMOTE Interpolates:
    Formula:  x_new = x_i + λ × (x_nn − x_i)    where λ ~ Uniform(0, 1)

    A random interpolation weight is drawn, populating sparse regions of the
    minority feature space to help classifiers learn a robust decision boundary.

  ▸ IMPORTANT: SMOTE is applied ONLY inside the training fold of each CV split.
    The test set remains untouched to reflect real-world imbalance.
""")

# Demonstrate SMOTE effect (on training data only — for visualization)
smote_demo = SMOTE(random_state=42, k_neighbors=5)
X_resampled_demo, y_resampled_demo = smote_demo.fit_resample(X_train, y_train)

print(f"  Before SMOTE (Training Set):")
print(f"    Class 0 (Legitimate): {(y_train == 0).sum():>6,}")
print(f"    Class 1 (Fraud):      {(y_train == 1).sum():>6,}")
print(f"\n  After SMOTE (Training Set):")
print(f"    Class 0 (Legitimate): {(y_resampled_demo == 0).sum():>6,}")
print(f"    Class 1 (Fraud):      {(y_resampled_demo == 1).sum():>6,}")
print(f"    Total samples:        {len(y_resampled_demo):>6,}")

# Plot SMOTE effect
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

before = [y_train.value_counts()[0], y_train.value_counts()[1]]
after = [(y_resampled_demo == 0).sum(), (y_resampled_demo == 1).sum()]

for ax, data, title in zip(axes, [before, after],
                            ["Before SMOTE", "After SMOTE"]):
    bars = ax.bar(["Legitimate", "Fraudulent"], data, color=colors,
                  edgecolor="black")
    for bar, val in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")

plt.suptitle("Effect of SMOTE on Class Balance (Training Set Only)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_smote_effect.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ Plot saved: plots/04_smote_effect.png")

# Clean up demo data
del X_resampled_demo, y_resampled_demo


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: THE API IMPERATIVE — sklearn vs imblearn Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 7: The API Imperative — sklearn vs imblearn Pipeline")
print("─" * 80)

print("""
  ▸ ❌ sklearn.pipeline.Pipeline  (FAILS with resampling)
    Standard pipeline steps expect a transform(X) method that only modifies
    the feature matrix. Resampling is ignored or crashes.

  ▸  imblearn.pipeline.Pipeline  (PRODUCTION STANDARD)
    Natively supports resampling. Modifies BOTH the feature matrix (X)
    and the target vector (y) strictly on the training fold via the
    fit_resample interface.

    Flow:  Split → SMOTE(X, y) → Train → Predict

  ▸ Pipeline Architecture (from the PDF):

    LR Pipeline:   StandardScaler → SMOTE → LogisticRegression
                   (LR is FATAL without scaler — regularization penalties
                    are distorted by massive transaction amount variances)

    RF Pipeline:   SMOTE → RandomForestClassifier
                   (NO scaler needed — tree-based models partition feature
                    space ordinally, rendering scale mathematically irrelevant)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CONSTRUCTING THE LINEAR PIPELINE (Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 8: Constructing the Linear Pipeline (Logistic Regression)")
print("─" * 80)

print("""
  Pipeline flow:
    1. StandardScaler — Standardization (μ=0, σ=1)
    2. SMOTE           — Synthetic Balancing
    3. LogisticRegression — Gradient Descent Scoring

  ⚠  StandardScaler MUST live INSIDE the pipeline. Scaling globally before
     splitting leaks mean and variance statistics from the test partition.
""")

# Build the LR pipeline (as specified in the PDF)
lr_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )),
])

# Train baseline LR pipeline
print("  Training baseline LR pipeline: StandardScaler → SMOTE → LR...")
t0 = time.time()
lr_pipeline.fit(X_train, y_train)
lr_time = time.time() - t0
print(f"  ✓ Trained in {lr_time:.2f}s")

# Evaluate on untouched test data
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
avg_prec_lr = average_precision_score(y_test, y_prob_lr)

print(f"\n  ▸ Logistic Regression (Baseline) — Test Set Evaluation:")
print(f"    ⚠  Accuracy DISCARDED. Using strict metrics only:")
print(f"    Precision:          {precision_lr:.4f}")
print(f"    Recall:             {recall_lr:.4f}")
print(f"    F1-Score:           {f1_lr:.4f}")
print(f"    ROC-AUC:            {roc_auc_lr:.4f}")
print(f"    Avg Precision (PR): {avg_prec_lr:.4f}")

print(f"\n  ▸ Classification Report:")
print(classification_report(y_test, y_pred_lr,
                            target_names=["Legitimate", "Fraud"], digits=4))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"  ▸ Confusion Matrix:")
print(f"    ┌───────────────┬──────────────┬──────────────┐")
print(f"    │               │ Pred Legit   │ Pred Fraud   │")
print(f"    ├───────────────┼──────────────┼──────────────┤")
print(f"    │ Actual Legit  │ TN={cm_lr[0,0]:>7,}  │ FP={cm_lr[0,1]:>5,}    │")
print(f"    │ Actual Fraud  │ FN={cm_lr[1,0]:>7,}  │ TP={cm_lr[1,1]:>5,}    │")
print(f"    └───────────────┴──────────────┴──────────────┘")
print(f"    → {cm_lr[1,1]} out of {cm_lr[1,0]+cm_lr[1,1]} fraud cases detected")
print(f"    → {cm_lr[0,1]} legitimate transactions falsely flagged")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CONSTRUCTING THE ENSEMBLE PIPELINE (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 9: Constructing the Ensemble Pipeline (Random Forest)")
print("─" * 80)

print("""
  Pipeline flow:
    1. SMOTE                    — Synthetic Balancing
    2. RandomForestClassifier   — Ensemble Tree Scoring

  ⚠  No StandardScaler required. Tree-based models partition feature space
     ordinally, rendering scale transformations mathematically irrelevant.
""")

# Build the RF pipeline (NO scaler, as specified in the PDF)
rf_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

# Train baseline RF pipeline
print("  Training baseline RF pipeline: SMOTE → RandomForest...")
t0 = time.time()
rf_pipeline.fit(X_train, y_train)
rf_time = time.time() - t0
print(f"  ✓ Trained in {rf_time:.2f}s")

# Evaluate on untouched test data
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
avg_prec_rf = average_precision_score(y_test, y_prob_rf)

print(f"\n  ▸ Random Forest (Baseline) — Test Set Evaluation:")
print(f"    Precision:          {precision_rf:.4f}")
print(f"    Recall:             {recall_rf:.4f}")
print(f"    F1-Score:           {f1_rf:.4f}")
print(f"    ROC-AUC:            {roc_auc_rf:.4f}")
print(f"    Avg Precision (PR): {avg_prec_rf:.4f}")

print(f"\n  ▸ Classification Report:")
print(classification_report(y_test, y_pred_rf,
                            target_names=["Legitimate", "Fraud"], digits=4))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"  ▸ Confusion Matrix:")
print(f"    ┌───────────────┬──────────────┬──────────────┐")
print(f"    │               │ Pred Legit   │ Pred Fraud   │")
print(f"    ├───────────────┼──────────────┼──────────────┤")
print(f"    │ Actual Legit  │ TN={cm_rf[0,0]:>7,}  │ FP={cm_rf[0,1]:>5,}    │")
print(f"    │ Actual Fraud  │ FN={cm_rf[1,0]:>7,}  │ TP={cm_rf[1,1]:>5,}    │")
print(f"    └───────────────┴──────────────┴──────────────┘")
print(f"    → {cm_rf[1,1]} out of {cm_rf[1,0]+cm_rf[1,1]} fraud cases detected")
print(f"    → {cm_rf[0,1]} legitimate transactions falsely flagged")

# Feature importance
feature_importance = pd.Series(
    rf_pipeline.named_steps["classifier"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(f"\n  ▸ Top 10 Most Important Features (Random Forest):")
for i, (feat, imp) in enumerate(feature_importance.head(10).items(), 1):
    bar = "█" * int(imp * 100)
    print(f"    {i:2d}. {feat:<8s} {imp:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 7))
top_15 = feature_importance.head(15)
bars = ax.barh(range(len(top_15)), top_15.values,
               color=sns.color_palette("viridis", len(top_15)),
               edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15.index, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Random Forest — Top 15 Feature Importances",
             fontsize=14, fontweight="bold")
for i, (val, bar) in enumerate(zip(top_15.values, bars)):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/05_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: THE ENGINE ROOM — GridSearchCV (Hyperparameter Tuning)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 10: The Engine Room — GridSearchCV (Hyperparameter Tuning)")
print("─" * 80)

print("""
  ▸ GridSearchCV safely applies SMOTE inside every fold calculation for
    every parameter combination, ensuring ZERO leakage during tuning.

  ▸ 5-Fold Cross Validation Flow:
    80%% Training Fold → SMOTE Chamber → Model Training
                                              ↓
    20%% Validation Fold ←←←← Model Prediction

    "Never expose your validation fold to scaling or resampling."

  ▸ Hyperparameter Grids (from PDF):
    SMOTE:   smote__k_neighbors = [3, 5, 7]
    LR:      classifier__C = [0.01, 0.1, 1.0]
    RF:      classifier__max_depth = [10, 20, None]
""")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ── 10.1: Tuning Logistic Regression Pipeline ──
print("  ── Tuning Logistic Regression Pipeline ──")
print("     Pipeline: StandardScaler → SMOTE → LogisticRegression")

lr_param_grid = {
    "smote__k_neighbors": [3, 5, 7],              # SMOTE hyperparameter (from PDF)
    "classifier__C": [0.01, 0.1, 1.0],            # LR regularization (from PDF)
    "classifier__penalty": ["l2"],
}

lr_grid = GridSearchCV(
    lr_pipeline,           # The imblearn pipeline (scaler → SMOTE → LR)
    param_grid=lr_param_grid,
    cv=cv,
    scoring="roc_auc",     # ← NOT accuracy!
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
lr_grid.fit(X_train, y_train)
lr_tune_time = time.time() - t0

print(f"  ✓ Grid search completed in {lr_tune_time:.2f}s")
print(f"    Total combinations tested: {len(lr_grid.cv_results_['params'])}")
print(f"    Best parameters:")
for param, val in lr_grid.best_params_.items():
    print(f"      {param}: {val}")
print(f"    Best CV ROC-AUC: {lr_grid.best_score_:.4f}")

# Evaluate tuned LR on test set
lr_tuned = lr_grid.best_estimator_
y_pred_lr_tuned = lr_tuned.predict(X_test)
y_prob_lr_tuned = lr_tuned.predict_proba(X_test)[:, 1]

precision_lr_t = precision_score(y_test, y_pred_lr_tuned)
recall_lr_t = recall_score(y_test, y_pred_lr_tuned)
f1_lr_t = f1_score(y_test, y_pred_lr_tuned)
roc_auc_lr_t = roc_auc_score(y_test, y_prob_lr_tuned)
avg_prec_lr_t = average_precision_score(y_test, y_prob_lr_tuned)

print(f"\n    Tuned LR — Test Results:")
print(f"      Precision:          {precision_lr_t:.4f}")
print(f"      Recall:             {recall_lr_t:.4f}")
print(f"      F1-Score:           {f1_lr_t:.4f}")
print(f"      ROC-AUC:            {roc_auc_lr_t:.4f}")
print(f"      Avg Precision (PR): {avg_prec_lr_t:.4f}")

print(f"\n    Classification Report (Tuned LR):")
print(classification_report(y_test, y_pred_lr_tuned,
                            target_names=["Legitimate", "Fraud"], digits=4))

# ── 10.2: Tuning Random Forest Pipeline ──
print("  ── Tuning Random Forest Pipeline ──")
print("     Pipeline: SMOTE → RandomForestClassifier (no scaler)")

rf_param_grid = {
    "smote__k_neighbors": [3, 5, 7],              # SMOTE hyperparameter (from PDF)
    "classifier__max_depth": [10, 20, None],       # From PDF grid
    "classifier__n_estimators": [100],             # Fixed for tractable computation
}

rf_grid = GridSearchCV(
    rf_pipeline,           # The imblearn pipeline (SMOTE → RF)
    param_grid=rf_param_grid,
    cv=cv,
    scoring="roc_auc",     # ← NOT accuracy!
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
rf_grid.fit(X_train, y_train)
rf_tune_time = time.time() - t0

print(f"  ✓ Grid search completed in {rf_tune_time:.2f}s")
print(f"    Total combinations tested: {len(rf_grid.cv_results_['params'])}")
print(f"    Best parameters:")
for param, val in rf_grid.best_params_.items():
    print(f"      {param}: {val}")
print(f"    Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

# Evaluate tuned RF on test set
rf_tuned = rf_grid.best_estimator_
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_prob_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]

precision_rf_t = precision_score(y_test, y_pred_rf_tuned)
recall_rf_t = recall_score(y_test, y_pred_rf_tuned)
f1_rf_t = f1_score(y_test, y_pred_rf_tuned)
roc_auc_rf_t = roc_auc_score(y_test, y_prob_rf_tuned)
avg_prec_rf_t = average_precision_score(y_test, y_prob_rf_tuned)

print(f"\n    Tuned RF — Test Results:")
print(f"      Precision:          {precision_rf_t:.4f}")
print(f"      Recall:             {recall_rf_t:.4f}")
print(f"      F1-Score:           {f1_rf_t:.4f}")
print(f"      ROC-AUC:            {roc_auc_rf_t:.4f}")
print(f"      Avg Precision (PR): {avg_prec_rf_t:.4f}")

print(f"\n    Classification Report (Tuned RF):")
print(classification_report(y_test, y_pred_rf_tuned,
                            target_names=["Legitimate", "Fraud"], digits=4))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MODEL COMPARISON & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 11: Model Comparison & Visualization")
print("─" * 80)

# 11.1 — Confusion Matrices (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, name, cmap in [
    (axes[0], y_pred_lr_tuned, "Logistic Regression (Tuned)", "Blues"),
    (axes[1], y_pred_rf_tuned, "Random Forest (Tuned)", "Greens"),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=",d", cmap=cmap, ax=ax,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                linewidths=1, linecolor="black", cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(name, fontsize=13, fontweight="bold")

plt.suptitle("Confusion Matrices — Tuned Models (Evaluated on Untouched Test Data)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/06_confusion_matrices.png")

# 11.2 — ROC Curves
fig, ax = plt.subplots(figsize=(9, 7))

for y_prob, label, color in [
    (y_prob_lr_tuned,
     f"LR: StandardScaler→SMOTE→LR (AUC={roc_auc_lr_t:.4f})", "#3498db"),
    (y_prob_rf_tuned,
     f"RF: SMOTE→RF (AUC={roc_auc_rf_t:.4f})", "#e74c3c"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=label)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
        label="Random Classifier (AUC=0.5)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve — Tuned Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "07_roc_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/07_roc_curves.png")

# 11.3 — Precision-Recall Curves
fig, ax = plt.subplots(figsize=(9, 7))

for y_prob, label, color in [
    (y_prob_lr_tuned,
     f"LR (AP={avg_prec_lr_t:.4f})", "#3498db"),
    (y_prob_rf_tuned,
     f"RF (AP={avg_prec_rf_t:.4f})", "#e74c3c"),
]:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color=color, linewidth=2.5, label=label)

baseline = y_test.sum() / len(y_test)
ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
           label=f"No-skill baseline ({baseline:.4f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — Tuned Models",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.set_xlim([0, 1.02])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "08_precision_recall_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/08_precision_recall_curves.png")

# 11.4 — Metric comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
lr_scores = [precision_lr_t, recall_lr_t, f1_lr_t, roc_auc_lr_t]
rf_scores = [precision_rf_t, recall_rf_t, f1_rf_t, roc_auc_rf_t]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width / 2, lr_scores, width,
               label="Logistic Regression (Tuned)",
               color="#3498db", edgecolor="black", linewidth=0.7)
bars2 = ax.bar(x + width / 2, rf_scores, width,
               label="Random Forest (Tuned)",
               color="#e74c3c", edgecolor="black", linewidth=0.7)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison — Key Metrics (NOT Accuracy)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "09_model_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/09_model_comparison.png")

# 11.5 — The Accuracy Trap visualization
fig, ax = plt.subplots(figsize=(10, 6))

models = ["Naive\n(All Legit)", "Logistic\nRegression", "Random\nForest"]
accuracies = [
    naive_acc,
    accuracy_score(y_test, y_pred_lr_tuned),
    accuracy_score(y_test, y_pred_rf_tuned),
]
recalls = [naive_rec, recall_lr_t, recall_rf_t]
roc_aucs = [0.5, roc_auc_lr_t, roc_auc_rf_t]

x = np.arange(len(models))
width = 0.25

b1 = ax.bar(x - width, accuracies, width,
            label="Accuracy (MISLEADING)", color="#95a5a6",
            edgecolor="black", hatch="//")
b2 = ax.bar(x, recalls, width,
            label="Recall (TRUE Performance)", color="#e74c3c",
            edgecolor="black")
b3 = ax.bar(x + width, roc_aucs, width,
            label="ROC-AUC", color="#3498db", edgecolor="black")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Trap #1: The Illusion of Accuracy — Why Accuracy is MISLEADING",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=10, loc="center right")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "10_accuracy_trap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Plot saved: plots/10_accuracy_trap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: THE ZERO-LEAKAGE PROTOCOL — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  SECTION 12: The Zero-Leakage Protocol — Summary")
print("─" * 80)

print("""
  ▸ Architectural Synthesis — The Imblearn Pipeline Loop:

    Raw Imbalanced Data   →  Stratified Split  →  [ Scaler → SMOTE → Model ]  →  GridSearchCV  →  Final Evaluation
    (99.83% Legit /           (Exact class          (if LR)   Balancing  Gradient     Tuning         (ROC-AUC &
     0.17% Fraud)              ratios)                                   Descent/    (Multi-fold     Confusion Matrix
                                                                         Ensemble     iteration)     on untouched
                                                                         Tree                        test data)
                                                                         Scoring)

   Ditch Accuracy. Optimize using Recall, F1, and ROC-AUC.
   Use SMOTE to interpolate and generate, NEVER just duplicate.
   NEVER apply SMOTE or Scalers prior to the Train/Test Split.
   ALWAYS use imblearn.pipeline.Pipeline to safely isolate resampling in CV.
   Tune preprocessing and model hyperparameters holistically inside GridSearchCV.

  → Deploy with precision.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  FINAL SUMMARY — Fraud Detection Pipeline Results")
print("=" * 80)

summary_data = {
    "Model": [
        "Naive (All Legitimate)",
        "LR Baseline (Scaler→SMOTE→LR)",
        "LR Tuned (GridSearchCV)",
        "RF Baseline (SMOTE→RF)",
        "RF Tuned (GridSearchCV)",
    ],
    "Precision": [naive_prec, precision_lr, precision_lr_t,
                  precision_rf, precision_rf_t],
    "Recall": [naive_rec, recall_lr, recall_lr_t,
               recall_rf, recall_rf_t],
    "F1-Score": [naive_f1, f1_lr, f1_lr_t,
                 f1_rf, f1_rf_t],
    "ROC-AUC": [0.5, roc_auc_lr, roc_auc_lr_t,
                roc_auc_rf, roc_auc_rf_t],
}

summary_df = pd.DataFrame(summary_data)
print(f"\n{summary_df.to_string(index=False)}")

# Determine the best model (excluding naive)
real_models = summary_df[summary_df["Model"] != "Naive (All Legitimate)"]
best_idx = real_models["ROC-AUC"].idxmax()
best_model = summary_df.loc[best_idx, "Model"]
best_roc = summary_df.loc[best_idx, "ROC-AUC"]
best_recall = summary_df.loc[best_idx, "Recall"]

print(f"\n   BEST MODEL: {best_model}")
print(f"     ROC-AUC: {best_roc:.4f}  |  Recall: {best_recall:.4f}")

# Save results
try:
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_results.csv")
except NameError:
    results_path = os.path.join(os.getcwd(), "model_comparison_results.csv")
summary_df.to_csv(results_path, index=False)
print(f"\n  ✓ Results saved to: model_comparison_results.csv")

print(f"""
  ▸ KEY TAKEAWAYS:
    1. SMOTE synthesized minority samples via interpolation (x_new = x_i + λ(x_nn - x_i)),
       enabling models to learn fraud patterns despite 0.17% fraud rate.

    2. The imblearn.pipeline.Pipeline ensured ZERO data leakage:
       - SMOTE applied ONLY during training (never on test/validation data)
       - StandardScaler fitted ONLY on training folds (no test statistics leaked)

    3. Two architecturally different pipelines were built:
       - LR: StandardScaler → SMOTE → LogisticRegression (scaler is FATAL for LR)
       - RF: SMOTE → RandomForestClassifier (no scaler — trees are scale-invariant)

    4. GridSearchCV tuned BOTH SMOTE k_neighbors AND classifier hyperparameters
       holistically, with StratifiedKFold preserving class ratios in every fold.

    5. Accuracy was DISCARDED and EXPOSED as misleading:
       - Naive "all-legitimate" achieves {naive_acc * 100:.2f}% accuracy → catches ZERO fraud
       - Precision, Recall, and ROC-AUC reveal the TRUE model performance.

    6. ROC-AUC target of 0.85+ was {"ACHIEVED ✓" if best_roc >= 0.85 else "not reached"}
       (Best: {best_roc:.4f}).
""")

print("=" * 80)
print("  PROJECT 2 COMPLETE ✓")
print("=" * 80)
