import pandas as pd

FILENAMES = [
    "DefenseDataCube_Transformed.csv",
    "KickingDataCube_Transformed.csv",
    "QBDataCube_Transformed.csv",
    "ReceivingDataCube_Transformed.csv",
    "RushingDataCube_Transformed.csv"
]

for filename in FILENAMES:

    print()
    print(filename)

    df = pd.read_csv(filename)

    player_column = df.columns[0]

    career_lengths = (
        df.groupby(
            player_column
        ).size()
    )

    bad_players = career_lengths[
        career_lengths != 100
    ]

    if len(bad_players) == 0:

        print(
            "SUCCESS: Every player "
            "has exactly 100 rows."
        )

    else:

        print(
            "ERROR:"
        )

        print(
            bad_players
        )
