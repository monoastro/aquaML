"""
train.py

Responsible for:

- Loading the dataset
- Creating the model
- Training
- Evaluating
- Saving the trained model
"""

import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import load_dataset

from config import (
    MODEL_NAME,
    MODEL_CLASS,
    MODEL_PARAMETERS,
    MODEL_SAVE_PATH,
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

X_train, X_test, y_train, y_test = load_dataset()


# ==========================================================
# Create Model
# ==========================================================

print("=" * 60)
print(f"Creating Model : {MODEL_NAME}")
print("=" * 60)

model = MODEL_CLASS(
    **MODEL_PARAMETERS
)


# ==========================================================
# Train
# ==========================================================

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.\n")


# ==========================================================
# Evaluate
# ==========================================================

print("=" * 60)
print("Evaluation")
print("=" * 60)

prediction = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print(f"Accuracy : {accuracy * 100:.2f}%")

print()

print("Classification Report")

print(
    classification_report(
        y_test,
        prediction
    )
)

print()

print("Confusion Matrix")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)

print()


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    MODEL_SAVE_PATH
)

print("=" * 60)
print("Training Finished")
print("=" * 60)

print()

print(f"Model saved to:")

print(MODEL_SAVE_PATH)

# ==========================================================
# Feature Importance (tree-based models only)
# ==========================================================

if hasattr(model, "feature_importances_"):

    print("\n" + "=" * 60)
    print("Feature Importance")
    print("=" * 60)

    for feature, importance in zip(
        X_train.columns,
        model.feature_importances_
    ):
        print(f"{feature:<20} {importance:.4f}")


# ==========================================================
# Decision Tree Structure (DecisionTreeClassifier only)
# ==========================================================

from sklearn.tree import DecisionTreeClassifier

if isinstance(model, DecisionTreeClassifier):

    from sklearn.tree import export_text

    print("\n" + "=" * 60)
    print("Learned Decision Tree")
    print("=" * 60)

    print(
        export_text(
            model,
            feature_names=list(X_train.columns)
        )
    )
