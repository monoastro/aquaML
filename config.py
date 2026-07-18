"""
config.py

Central configuration for the AquaEdge ML project.

To switch models, change MODEL_NAME to any key in MODEL_CONFIG.
Everything else (training, inference, serial monitor) picks it up automatically.
"""

from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Project Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Dataset
DATASET_PATH = DATA_DIR / "kenya_water.csv"
FEATURE_COLUMNS = [
    "pH",
    "TDS_ppm",
    "Turbidity_NTU",
    "Temperature_C",
]
LABEL_COLUMN = "Label"


# Dataset Split Parameters
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
SHUFFLE_DATASET = True


# Active Model
# Change this ONE line to switch between any model below.
MODEL_NAME = "DecisionTree"


# ==========================================================
# Model Registry
#
# Each entry defines:
#   class      - sklearn estimator class
#   parameters - constructor keyword arguments
#   save_path  - where the .pkl lives (train.py writes here,
#                inference / serial_monitor load from here)
# ==========================================================

MODEL_CONFIG = {

    # ----------------------------------------------------------
    # Decision Tree  (string labels: Safe / Unsafe)
    # ----------------------------------------------------------
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "parameters": {
            "criterion":         "gini",
            "max_depth":         5,
            "min_samples_split": 10,
            "random_state":      RANDOM_STATE,
        },
        "save_path": MODEL_DIR / "decision_tree.pkl",
    },

    # ----------------------------------------------------------
    # Decision Tree Balanced  (int labels: 0 / 1)
    # ----------------------------------------------------------
    "DecisionTreeBalanced": {
        "class": DecisionTreeClassifier,
        "parameters": {
            "criterion":         "gini",
            "max_depth":         10,
            "min_samples_split": 2,
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
        },
        "save_path": MODEL_DIR / "decision_tree_model.pkl",
    },

    # ----------------------------------------------------------
    # Random Forest  (string labels: Safe / Unsafe)
    # ----------------------------------------------------------
    "RandomForest": {
        "class": RandomForestClassifier,
        "parameters": {
            "n_estimators":      100,
            "criterion":         "gini",
            "max_depth":         10,
            "min_samples_split": 10,
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        },
        "save_path": MODEL_DIR / "random_forest.pkl",
    },

    # ----------------------------------------------------------
    # Random Forest Balanced  (int labels: 0 / 1)
    # ----------------------------------------------------------
    "RandomForestBalanced": {
        "class": RandomForestClassifier,
        "parameters": {
            "n_estimators":      100,
            "criterion":         "gini",
            "max_depth":         10,
            "min_samples_split": 2,
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
        },
        "save_path": MODEL_DIR / "random_forest_model.pkl",
    },

    # ----------------------------------------------------------
    # KNN  (int labels: 0 / 1)
    # ----------------------------------------------------------
    "KNN": {
        "class": KNeighborsClassifier,
        "parameters": {
            "n_neighbors": 5,
            "weights":     "uniform",
            "algorithm":   "auto",
            "metric":      "minkowski",
            "p":           2,
        },
        "save_path": MODEL_DIR / "knn.pkl",
    },

    # ----------------------------------------------------------
    # SVM  (int labels: 0 / 1)
    # ----------------------------------------------------------
    "SVM": {
        "class": SVC,
        "parameters": {
            "C":            1.0,
            "kernel":       "rbf",
            "gamma":        "scale",
            "class_weight": "balanced",
            "probability":  True,
            "random_state": RANDOM_STATE,
        },
        "save_path": MODEL_DIR / "svm_model.pkl",
    },

}


# Convenience Variables  (derived from MODEL_NAME above)
ACTIVE_MODEL     = MODEL_CONFIG[MODEL_NAME]
MODEL_CLASS      = ACTIVE_MODEL["class"]
MODEL_PARAMETERS = ACTIVE_MODEL["parameters"]
MODEL_SAVE_PATH  = ACTIVE_MODEL["save_path"]


# Serial / ESP32
SERIAL_PORT      = "/dev/ttyUSB0"   # change to match your system
SERIAL_BAUD_RATE = 115200
SERIAL_TIMEOUT   = 2                # seconds
