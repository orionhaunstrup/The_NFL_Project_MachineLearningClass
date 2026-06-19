import pandas as pd

FILENAMES = [
    "DefenseDataCube_Transformed2.csv",
    "KickingDataCube_Transformed2.csv",
    "QBDataCube_Transformed2.csv",
    "ReceivingDataCube_Transformed2.csv",
    "RushingDataCube_Transformed2.csv"
]

for filename in FILENAMES:

    print()
    print(f"Processing {filename}")

    df = pd.read_csv(filename)

    player_column = df.columns[0]

    player_vectors = []

    player_names = []

    for player_name, player_df in (
            df.groupby(
                player_column,
                sort=False
            )
            ):

        player_names.append(
            player_name
        )

        #
        # Remove:
        # column 0 = PLAYER NAME
        # column 1 = YEAR
        #

        values = (
            player_df
            .iloc[:, 2:]
            .to_numpy()
            .flatten()
            .tolist()
        )

        player_vectors.append(
            values
        )

    output_df = pd.DataFrame(
        player_vectors
    )

    output_df.insert(
        0,
        "PLAYER NAME",
        player_names
    )

    output_filename = (
        filename.replace(
            ".csv",
            "_PCAReady.csv"
        )
    )

    output_df.to_csv(
        output_filename,
        index=False
    )

    print(
        f"Saved {output_filename}"
    )

    print(
        f"Players: {len(player_names)}"
    )

    print(
        f"Features per player: "
        f"{len(player_vectors[0])}"
    )

print()
print("Done.")
