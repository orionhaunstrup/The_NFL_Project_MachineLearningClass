import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATACUBES = {
    "QB": "QBDataCube.csv",
    "Rushing": "RushingDataCube.csv",
    "Receiving": "ReceivingDataCube.csv",
    "Defense": "DefenseDataCube.csv",
    "Kicking": "KickingDataCube.csv"
}


## This line of code accounts for the fact that sometimes we'll have
## a position (QB, Kicker, etc) and num years pair that leaves us with
## opnly a small number of players who meet that criteria. In that case
## it's a little silly to try to use clustering. When this happens,
## we'll just leave that tiny group as a small cluster.
MIN_PLAYERS_FOR_CLUSTERING = 10


def build_non_jagged_cube(df, years):
    ## This function tryies to construct a non-jagged new datacube
    ## for a given position and num years of training data.
    ##
    ## It removes players with fewer than 'years' number of seasons.

    
    # First let's use df groupby to sort by player career length
    career_lengths = df.groupby("PLAYER NAME").size()

    # Next let's keep only players with at least 'years' seasons
    valid_players = career_lengths[career_lengths >= years].index

    df_filtered = df[df["PLAYER NAME"].isin(valid_players)]
    player_vectors = []

    for player_name, player_data in df_filtered.groupby("PLAYER NAME"):
        # Trim off the extra seasons past those we're examining
        player_data = player_data.iloc[:years]
        # Drop the extra columns
        player_data = player_data.drop(
            columns=["PLAYER NAME", "YEAR", "HOWGOOD RAW SCORE", "PERCENTILE"])
        # Flatten the seasons into a single vector
        vector = player_data.to_numpy().flatten()
        player_vectors.append(vector)

    return np.array(player_vectors)


for cube_name, filename in DATACUBES.items():

    print()
    print(cube_name)
    print()

    df = pd.read_csv(filename)

    for z in range(3, 11):

        print(f"\nStarting {cube_name}_{z}year...")
        X = build_non_jagged_cube(df, z)
        n_players = len(X)

        # Run this if there are too few players to cluster meaningfully
        if n_players < MIN_PLAYERS_FOR_CLUSTERING:

            print(
                f"{cube_name}_{z}year -> "
                f"Players={n_players}, "
                f"Best k=1 (insufficient data)"
            )
            continue

        max_k = min(60, n_players - 1)
        if max_k < 5:
            print(
                f"{cube_name}_{z}year -> "
                f"Players={n_players}, "
                f"Best k=1"
            )
            continue

        best_k = None
        best_score = -1

        k_values = list(range(2, max_k + 1))
        total_k = len(k_values)

        ## Let's have the program run a small progress bar
        ## so we don't lose our minds waiting the whole time
        ## without knowing how much further there is to go
        progress_markers = {
            max(1, int(total_k * 0.25)): "25%",
            max(1, int(total_k * 0.50)): "50%",
            max(1, int(total_k * 0.75)): "75%",
            total_k: "100%"
        }

        for idx, k in enumerate(k_values, start=1):

            if idx in progress_markers:
                print(
                    f"{cube_name}_{z}year: "
                    f"{progress_markers[idx]} complete"
                )

            try:

                kmeans = KMeans(
                    n_clusters=k,
                    init="random",
                    n_init=10,
                    random_state=42
                )

                labels = kmeans.fit_predict(X)

                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:

                print(
                    f"Error on {cube_name}_{z}year "
                    f"k={k}: {e}"
                )

        print(
            f"{cube_name}_{z}year -> "
            f"Players={n_players}, "
            f"Best k={best_k}, "
            f"Silhouette={best_score:.4f}"
        )

print("\nAll the experiments are complete.")
