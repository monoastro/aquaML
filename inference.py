"""
inference.py

Run inference using a previously trained model.
"""

from model_manager import ModelManager


def predict_water():

    manager = ModelManager()

    manager.load()

    print("=" * 60)
    print("AquaEdge Water Quality Prediction")
    print("=" * 60)

    while True:

        print()

        try:

            ph = float(input("pH: "))
            tds = float(input("TDS (ppm): "))
            turbidity = float(input("Turbidity (NTU): "))
            temperature = float(input("Temperature (°C): "))

        except ValueError:

            print()

            print("Please enter numeric values.")

            continue

        result = manager.classify(

            ph=ph,

            tds=tds,

            turbidity=turbidity,

            temperature=temperature

        )

        print()

        print("=" * 60)

        print(
            f"Prediction : {result['prediction']}"
        )

        print()

        print("Confidence")

        for label, probability in result["confidence"].items():

            print(
                f"{label:10}: {probability * 100:.2f}%"
            )

        print("=" * 60)

        again = input("\nPredict another sample? (y/n): ")

        if again.lower() != "y":

            break


if __name__ == "__main__":

    predict_water()
