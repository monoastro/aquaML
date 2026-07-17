"""
dataset.py

Responsible for:

- Loading the dataset
- Validating dataset structure
- Cleaning the dataset
- Splitting into train/test sets

No machine learning happens here.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    SHUFFLE_DATASET
)


class Dataset:

    def __init__(self):

        self.dataframe = None

    # =====================================================
    # Load Dataset
    # =====================================================

    def load(self):

        print("Loading dataset...")

        self.dataframe = pd.read_csv(DATASET_PATH)

        print(f"Loaded {len(self.dataframe)} samples.")

    # =====================================================
    # Validate Dataset
    # =====================================================

    def validate(self):

        print("Validating dataset...")

        required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

        for column in required_columns:

            if column not in self.dataframe.columns:

                raise ValueError(
                    f"Missing required column: {column}"
                )

        print("Dataset validation passed.")

    # =====================================================
    # Clean Dataset
    # =====================================================

    def clean(self):

        print("Cleaning dataset...")

        before = len(self.dataframe)

        self.dataframe = self.dataframe.dropna()

        after = len(self.dataframe)

        print(
            f"Removed {before-after} rows containing missing values."
        )

    # =====================================================
    # Shuffle Dataset
    # =====================================================

    def shuffle(self):

        if not SHUFFLE_DATASET:

            return

        print("Shuffling dataset...")

        self.dataframe = self.dataframe.sample(

            frac=1,

            random_state=RANDOM_STATE

        ).reset_index(drop=True)

    # =====================================================
    # Dataset Summary
    # =====================================================

    def summary(self):

        print()

        print("=" * 60)

        print("Dataset Summary")

        print("=" * 60)

        print()

        print(f"Samples : {len(self.dataframe)}")

        print()

        print("Features:")

        for feature in FEATURE_COLUMNS:

            print(f"  - {feature}")

        print()

        print("Class Distribution")

        print()

        print(
            self.dataframe[LABEL_COLUMN].value_counts()
        )

        print()

    # =====================================================
    # Split Dataset
    # =====================================================

    def split(self):

        X = self.dataframe[
            FEATURE_COLUMNS
        ]

        y = self.dataframe[
            LABEL_COLUMN
        ]

        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE

        )


# ==========================================================
# Convenience Function
# ==========================================================

def load_dataset():

    dataset = Dataset()

    dataset.load()

    dataset.validate()

    dataset.clean()

    dataset.shuffle()

    dataset.summary()

    return dataset.split()
