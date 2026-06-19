import math
import pandas as pd

TARGET_LENGTH = 100

FILENAMES = [
    "DefenseDataCube_Transformed1.csv",
    "KickingDataCube_Transformed1.csv",
    "QBDataCube_Transformed1.csv",
    "ReceivingDataCube_Transformed1.csv",
    "RushingDataCube_Transformed1.csv"
]

for filename in FILENAMES:

    print(f"Processing {filename}")

    df = pd.read_csv(filename)

    player_column = df.columns[0]

    transformed_rows = []

    for player_name, player_df in (
            df.groupby(player_column,
                       sort=False)
            ):

        rows = (
            player_df
            .values
            .tolist()
        )

        career_length = len(rows)

        multiplier = math.ceil(
            TARGET_LENGTH
            / career_length
        )

        stretched_rows = []

        for row in rows:

            for _ in range(
                    multiplier
                    ):

                stretched_rows.append(
                    row.copy()
                )

        stretched_rows = (
            stretched_rows[
                :TARGET_LENGTH
            ]
        )

        transformed_rows.extend(
            stretched_rows
        )

    new_df = pd.DataFrame(
        transformed_rows,
        columns=df.columns
    )

    output_filename = (
        filename.replace(
            ".csv",
            "_Transformed2.csv"
        )
    )

    new_df.to_csv(
        output_filename,
        index=False
    )

    print(
        f"Saved {output_filename}"
    )

print()
print("Done.")
