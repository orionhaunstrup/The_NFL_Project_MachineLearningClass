import os
import pandas as pd

# Folder containing the five datacubes
DATA_FOLDER = "."

# The five datacube filenames
files = [
    "DefenseDataCube.csv",
    "KickingDataCube.csv",
    "QBDataCube.csv",
    "ReceivingDataCube.csv",
    "RushingDataCube.csv"
]

for filename in files:

    filepath = os.path.join(DATA_FOLDER, filename)

    # Read the datacube
    df = pd.read_csv(filepath)

    # Keep only the first and last columns
    df = df.iloc[:, [0, -1]]

    # Overwrite the original file
    df.to_csv(filepath, index=False)

    print(f"Trimmed {filename}")

print("\nDone! All five datacubes now contain only:")
print("  • PLAYER NAME")
print("  • PERCENTILE")
