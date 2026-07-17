"""
model_manager.py
Responsible for:
- Creating machine learning models
- Saving trained models
- Loading trained models
- Running inference
"""

import joblib
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    MODEL_CLASS,
    MODEL_PARAMETERS,
    MODEL_SAVE_PATH,
)


class ModelManager:

    def __init__(self):

        self.model = None

    # ======================================================
    # Create Model
    # ======================================================

    def create(self):

        self.model = MODEL_CLASS(
            **MODEL_PARAMETERS
        )

    # ======================================================
    # Train Model
    # ======================================================

    def train(
        self,
        X_train,
        y_train
    ):

        self.model.fit(
            X_train,
            y_train
        )

    # ======================================================
    # Predict
    # ======================================================

    def predict(self, X):

        return self.model.predict(X)

    # ======================================================
    # Predict Probabilities
    # ======================================================

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    # ======================================================
    # Save Model
    # ======================================================

    def save(self):

        joblib.dump(
            self.model,
            MODEL_SAVE_PATH
        )

    # ======================================================
    # Load Model
    # ======================================================

    def load(self):

        self.model = joblib.load(
            MODEL_SAVE_PATH
        )

    # ======================================================
    # Interactive Prediction
    # ======================================================

    def classify(
        self,
        ph,
        tds,
        turbidity,
        temperature
    ):

        sample = pd.DataFrame(

            [[
                ph,
                tds,
                turbidity,
                temperature
            ]],

            columns=FEATURE_COLUMNS

        )

        prediction = self.model.predict(
            sample
        )[0]

        probabilities = self.model.predict_proba(
            sample
        )[0]

        confidence = dict(

            zip(

                self.model.classes_,

                probabilities

            )

        )

        return {

            "prediction": prediction,

            "confidence": confidence

        }
