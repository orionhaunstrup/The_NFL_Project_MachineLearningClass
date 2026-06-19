import pandas as pd

FILENAMES = [
    "DefenseDataCube.csv",
    "KickingDataCube.csv",
    "QBDataCube.csv",
    "ReceivingDataCube.csv",
    "RushingDataCube.csv"
]

for filename in FILENAMES:

    print(f"Processing {filename}")

    df = pd.read_csv(filename)

    # Remove the rightmost two columns
    df = df.iloc[:, :-2]

    # Count seasons for each player
    career_lengths = (
        df.groupby(
            df.columns[0]
        ).size()
    )

    # Add new column
    df["CareerLengthx5"] = (
        df[df.columns[0]]
        .map(career_lengths)
        * 5
    )

    output_filename = (
        filename.replace(
            ".csv",
            "_PCA.csv"
        )
    )

    df.to_csv(
        output_filename,
        index=False
    )

    print(
        f"Saved {output_filename}"
    )

print()
print("Done.")
