"""
serial_monitor.py

Continuously reads sensor data from an ESP32 over serial and
classifies each reading as Safe or Unsafe.

Expected ESP32 output format (one line per reading):
    pH,TDS,Turbidity,Temperature
    e.g.  7.12,320.5,1.8,26.4

Usage:
    python serial_monitor.py           # real ESP32
    python serial_monitor.py --mock    # simulated data (no hardware needed)

Configuration (port, baud rate, model) lives in config.py.
"""

import sys
import warnings

# Suppress sklearn version-mismatch warnings when loading pkl files
# that were saved with a different scikit-learn version.
warnings.filterwarnings(
    "ignore",
    message=".*InconsistentVersionWarning.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator.*",
    category=UserWarning,
)
import time
import random
import datetime
import argparse
import joblib
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    SERIAL_PORT,
    SERIAL_BAUD_RATE,
    SERIAL_TIMEOUT,
)


# ==========================================================
# Helpers
# ==========================================================

SEPARATOR = "=" * 60


def timestamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


# ==========================================================
# Mock ESP32 Data Generator
#
# Generates realistic water sensor readings based on the
# Kenya water dataset value ranges.
# Half the readings are intentionally "bad" to show both outcomes.
# ==========================================================

# Realistic ranges derived from the Kenya water dataset
_SAFE_RANGES = {
    "pH":            (6.5,  8.5),
    "TDS_ppm":       (28.6, 500.0),
    "Turbidity_NTU": (0.0,  4.0),
    "Temperature_C": (16.6, 31.5),
}

_UNSAFE_RANGES = {
    "pH":            (6.09, 9.96),   # out-of-range pH
    "TDS_ppm":       (500,  2806.0), # high dissolved solids
    "Turbidity_NTU": (4.0,  72.75),  # cloudy / turbid
    "Temperature_C": (16.6, 31.5),   # temperature alone rarely flags unsafe
}


def _rand(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


def generate_mock_reading(force_unsafe: bool = False) -> str:
    """Return a comma-separated sensor line like the ESP32 would send."""

    # Randomly decide safe vs unsafe (40 / 60 split to reflect real data)
    use_unsafe = force_unsafe or (random.random() < 0.60)
    ranges = _UNSAFE_RANGES if use_unsafe else _SAFE_RANGES

    ph          = _rand(*ranges["pH"])
    tds         = _rand(*ranges["TDS_ppm"])
    turbidity   = _rand(*ranges["Turbidity_NTU"])
    temperature = _rand(*ranges["Temperature_C"])

    return f"{ph},{tds},{turbidity},{temperature}"


# ==========================================================
# Serial Monitor
# ==========================================================

class SerialMonitor:

    def __init__(self, port: str, baud: int, timeout: float, mock: bool):
        self.port    = port
        self.baud    = baud
        self.timeout = timeout
        self.mock    = mock
        self.conn    = None
        self.model   = None

    # ----------------------------------------------------------
    # Load Model
    # ----------------------------------------------------------

    def load_model(self) -> None:

        if not MODEL_SAVE_PATH.exists():
            print(f"[ERROR] No saved model found at: {MODEL_SAVE_PATH}")
            print(f"        Train a model first:  python train.py")
            sys.exit(1)

        print(f"Loading model  : {MODEL_NAME}")
        print(f"Model path     : {MODEL_SAVE_PATH}")

        self.model = joblib.load(MODEL_SAVE_PATH)

        print("Model loaded successfully.\n")

    # ----------------------------------------------------------
    # Connect to Serial Port (real hardware)
    # ----------------------------------------------------------

    def connect(self) -> None:

        import serial
        import serial.tools.list_ports

        available = [p.device for p in serial.tools.list_ports.comports()]

        if not available:
            print("[ERROR] No serial ports detected.")
            print("        Make sure the ESP32 is connected via USB.")
            sys.exit(1)

        if self.port not in available:
            print(f"[WARNING] Configured port '{self.port}' not found.")
            print(f"          Available ports: {', '.join(available)}")
            print(f"          Update SERIAL_PORT in config.py, or use one of the above.")
            sys.exit(1)

        print(f"Connecting     : {self.port}  @  {self.baud} baud")

        self.conn = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout,
        )

        # Allow ESP32 time to reset after DTR toggle
        time.sleep(2)
        self.conn.reset_input_buffer()

        print("Connected.\n")

    # ----------------------------------------------------------
    # Parse a raw serial line into a feature DataFrame
    # ----------------------------------------------------------

    def parse_line(self, raw: str) -> pd.DataFrame | None:
        """
        Expects comma-separated floats in the order defined by
        FEATURE_COLUMNS:  pH, TDS_ppm, Turbidity_NTU, Temperature_C
        """

        parts = raw.strip().split(",")

        if len(parts) != len(FEATURE_COLUMNS):
            return None

        try:
            values = [float(p) for p in parts]
        except ValueError:
            return None

        return pd.DataFrame([values], columns=FEATURE_COLUMNS)

    # ----------------------------------------------------------
    # Classify a single sample
    # ----------------------------------------------------------

    def classify(self, sample: pd.DataFrame) -> dict:

        # Some models were fitted with named columns (DataFrame),
        # others with plain arrays. Pass the right type to avoid warnings.
        if hasattr(self.model, "feature_names_in_"):
            X = sample          # fitted with feature names -> keep DataFrame
        else:
            X = sample.values   # fitted without -> pass plain array

        label = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        confidence = dict(zip(self.model.classes_, probabilities))

        return {
            "label":      label,
            "confidence": confidence,
        }

    # ----------------------------------------------------------
    # Print a classification result
    # ----------------------------------------------------------

    def print_result(self, sample: pd.DataFrame, result: dict) -> None:

        label = result["label"]

        # Labels may be strings ("Safe"/"Unsafe") or ints (1/0)
        # depending on how the model was originally trained.
        label_str = str(label).lower()
        safe = label_str in ("safe", "1", "good", "potable")

        status = "  SAFE  " if safe else "  UNSAFE"

        print(f"\n[{timestamp()}]  {status}  ({label})")

        row = sample.iloc[0]
        print(
            f"  pH={row['pH']:.2f}  "
            f"TDS={row['TDS_ppm']:.1f} ppm  "
            f"Turbidity={row['Turbidity_NTU']:.2f} NTU  "
            f"Temp={row['Temperature_C']:.1f} °C"
        )

        conf_parts = [
            f"{cls}: {prob * 100:.1f}%"
            for cls, prob in result["confidence"].items()
        ]
        print("  Confidence  →  " + "   ".join(conf_parts))

    # ----------------------------------------------------------
    # Mock Loop  (no hardware)
    # ----------------------------------------------------------

    def run_mock(self, interval: float = 2.0) -> None:

        print(f"Generating a new reading every {interval:.0f}s.")
        print("Press  Ctrl+C  to stop.\n")

        reading_count = 0

        try:
            while True:

                raw = generate_mock_reading()

                sample = self.parse_line(raw)

                if sample is None:
                    continue

                reading_count += 1
                result = self.classify(sample)
                self.print_result(sample, result)

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\nStopped.  Total readings classified: {reading_count}")

    # ----------------------------------------------------------
    # Real Serial Loop
    # ----------------------------------------------------------

    def run_serial(self) -> None:

        print("Press  Ctrl+C  to stop.\n")

        skipped = 0

        try:
            while True:

                raw = self.conn.readline().decode("utf-8", errors="ignore")

                if not raw.strip():
                    continue

                sample = self.parse_line(raw)

                if sample is None:
                    skipped += 1
                    print(f"[{timestamp()}]  ESP32 > {raw.strip()}")
                    continue

                result = self.classify(sample)
                self.print_result(sample, result)

        except KeyboardInterrupt:
            print(f"\n\nStopped.  ({skipped} non-data lines skipped)")

        finally:
            if self.conn and self.conn.is_open:
                self.conn.close()
                print("Serial port closed.")

    # ----------------------------------------------------------
    # Entry Point
    # ----------------------------------------------------------

    def run(self, interval: float = 2.0) -> None:

        print(SEPARATOR)
        print("AquaEdge  —  Water Quality Monitor")
        print(SEPARATOR)
        print(f"Model          : {MODEL_NAME}")

        if self.mock:
            print(f"Mode           : MOCK  (simulated ESP32 data)")
        else:
            print(f"Mode           : SERIAL  ({self.port}  @  {self.baud} baud)")

        print(f"Features       : {', '.join(FEATURE_COLUMNS)}")
        print(SEPARATOR)

        if self.mock:
            self.run_mock(interval=interval)
        else:
            self.connect()
            self.run_serial()


# ==========================================================
# CLI
# ==========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="AquaEdge water quality monitor — ESP32 serial or mock mode."
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate simulated sensor data instead of reading from serial.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Seconds between mock readings (default: 2).",
    )

    parser.add_argument(
        "--port",
        type=str,
        default=SERIAL_PORT,
        metavar="PORT",
        help=f"Serial port to use (default: {SERIAL_PORT}).",
    )

    return parser.parse_args()


# ==========================================================
# Entry Point
# ==========================================================

def main() -> None:

    args = parse_args()

    monitor = SerialMonitor(
        port    = args.port,
        baud    = SERIAL_BAUD_RATE,
        timeout = SERIAL_TIMEOUT,
        mock    = args.mock,
    )

    monitor.load_model()
    monitor.run(interval=args.interval)


if __name__ == "__main__":
    main()
