""" The NFL Project - Machine Learning Class - Module 2

This version will introduce clustering

Orion Haunstrup
Summer 2026
"""


global RANDOMSEED
RANDOMSEED = 333



from collections import defaultdict
import csv
import random
import numpy as np
import statistics
import math
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


## First let's make a few dictionaries of useful info we'll use through
## out this code

DATACUBES = {
    "QB": "QBDataCube.csv",
    "Rushing": "RushingDataCube.csv",
    "Receiving": "ReceivingDataCube.csv",
    "Defense": "DefenseDataCube.csv",
    "Kicking": "KickingDataCube.csv"
}

PREFERRED_ORDER = ['A', 'B', 'C', 'D', 'F', 'Retired']




## Next, let's load all the data and get ready for sampling


def build_master_test_list(datacube_filename):
    ## The next two functions get the full list of all 'valid' players we
    ## Can select for study. Tom Brady after 3 years, TB after 7 years.
    ## It's all players who played for at least 3 seasons, at each
    ## point in their career
    
    player_rows = defaultdict(list)
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                player_rows[row[0]].append(row)
                
    master_test_list = []

    for player, rows in player_rows.items():
        career_length = len(rows)
        for years_to_reveal in range(3, career_length + 1):
            master_test_list.append(
                (player, years_to_reveal)
            )
            
    return master_test_list


def build_player_pool():
    ## This function gathers a list of all valid players,
    ## which data cube they're in, and how long they played for
    
    all_players = []
    
    for datacube_filename in DATACUBES.values():
        player_rows = defaultdict(list)
        with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            
            for row in reader:
                if row:
                    player_rows[row[0]].append(row)
                    
        for player, rows in player_rows.items():
            all_players.append((datacube_filename, player, len(rows)))

    return all_players


def distance_list_of_lists(list_of_lists1, list_of_lists2):
    ## This function calculates the measure of the distance between
    ## two vectors, using the standard Euclidean distance metric

    if len(list_of_lists1) != len(list_of_lists2):
        raise ValueError("Career lengths differ.")

    if len(list_of_lists1[0]) != len(list_of_lists2[0]):
        raise ValueError("Stat vector lengths differ.")

    total = 0

    for i in range(len(list_of_lists1)):
        for j in range(len(list_of_lists1[0])):

            total += (
                float(list_of_lists1[i][j])
                - float(list_of_lists2[i][j])
            ) ** 2

    return total


def find_N_smallest_indices(py_list, N):
    arr = np.array(py_list)
    if len(arr) == 0:
        return []
    N = min(N, len(arr))
    smallest_indices = np.argpartition(arr, N - 1)[:N]
    smallest_indices = smallest_indices[np.argsort(arr[smallest_indices])]
    return smallest_indices



def get_all_player_careers(datacube_filename):
    ## This function gathers every player's full career percentile vector
    ## For Example:
    ## {
    ##     "Tom Brady": [0.51, 0.63, 0.72, ...],
    ##     "Peyton Manning": [0.61, 0.68, 0.79, ...]
    ## }

    player_careers = defaultdict(list)
    
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row:
                player = row[0]
                percentile = float(row[-1])
                player_careers[player].append(percentile)

    return dict(player_careers)



def pad_careers_for_clustering(player_careers):
    ## KMeans requires every vector to be the same length.
    ## We pad shorter careers with -1's.

    max_career_length = max(len(career) for career in player_careers.values())
    player_names = []
    career_matrix = []

    for player, career in player_careers.items():

        padded_career = (career + [-1] * (max_career_length - len(career)))
        player_names.append(player)
        career_matrix.append(padded_career)

    return (player_names, career_matrix)



def build_clusters(datacube_filename, K):
    ## This function builds all the relevant clusters out of
    ## all NFL players who have ever played

    player_careers = (get_all_player_careers(datacube_filename))
    player_names, career_matrix = (pad_careers_for_clustering(player_careers))
    kmeans = KMeans(n_clusters=K, random_state=RANDOMSEED, n_init=10)

    cluster_labels = kmeans.fit_predict(career_matrix)

    return (player_names, cluster_labels, kmeans)



def rank_clusters_for_player(chosen_player_percentiles, years_revealed, kmeans):

    cluster_distances = []

    for cluster_num, centroid in enumerate(kmeans.cluster_centers_):

        distance = 0

        for year_idx in range(years_revealed):

            distance += (
                chosen_player_percentiles[year_idx] - centroid[year_idx]) ** 2

        cluster_distances.append((cluster_num, distance))

    cluster_distances.sort(key=lambda x: x[1])

    ranked_clusters = [
        cluster_num
        for cluster_num, distance
        in cluster_distances
    ]

    return ranked_clusters



def predict_num_future_seasons_from_clusters(ranked_clusters, chosen_player,
        player_names, cluster_labels, player_careers, years_revealed):

    for cluster_num in ranked_clusters:

        future_seasons_list = []

        for player, cluster in zip(player_names, cluster_labels):

            if cluster != cluster_num:
                continue
            if player == chosen_player:
                continue
            career = player_careers[player]
            future_seasons = (len(career) - years_revealed)

            if future_seasons >= 0:

                future_seasons_list.append(future_seasons)

        if len(future_seasons_list) > 0:

            return math.ceil(statistics.median(future_seasons_list))

    return None


def predict_future_percentiles_from_clusters(ranked_clusters, chosen_player,
        player_names, cluster_labels, player_careers, years_revealed,
        predicted_future_seasons):

    # Immediate retirement prediction.
    if predicted_future_seasons == 0:
        return []

    surviving_future_careers = []

    for cluster_num in ranked_clusters:

        surviving_future_careers = []

        for player, cluster in zip(player_names, cluster_labels):

            if cluster != cluster_num:
                continue

            if player == chosen_player:
                continue

            career = player_careers[player]

            future_career = career[years_revealed:]

            if len(future_career) >= (predicted_future_seasons):

                surviving_future_careers.append(future_career)

        # Stop at the first cluster
        # containing usable players.
        if len(surviving_future_careers) > 0:
            break

    if len(surviving_future_careers) == 0:

        return []

    predictions = []
    for future_year in range(predicted_future_seasons):
        values = []
        for career in surviving_future_careers:
            values.append(career[future_year])
        predictions.append(statistics.median(values))
    return predictions



def get_chosen_player_percentiles(
        chosen_player,
        datacube_filename):

    percentiles = []

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            if row[0] == chosen_player:

                percentiles.append(
                    float(row[-1])  # The already-in-the-datacube percentile
                )

    return percentiles


def score_prediction(predicted_percentiles, actual_percentiles):
    ## This function is brand new to the projec and lets me compare
    ## the machine's prediction to the actual rest of that player's
    ## career and see how accurate the prediction was!

    retirement_error = abs(
        len(predicted_percentiles)- len(actual_percentiles))

    overlap = min(len(predicted_percentiles), len(actual_percentiles))

    if overlap == 0:
        prediction_error = None

    else:

        differences = []
        for i in range(overlap):
            differences.append(
                abs(predicted_percentiles[i] - actual_percentiles[i]))
        prediction_error = (sum(differences) / len(differences))

    return retirement_error, prediction_error



def main_dendrogram_recent_qbs():

    datacube_filename = DATACUBES["QB"]

    YEAR_COLUMN = 1
    NUM_QBS = 40

    player_rows = defaultdict(list)

    print("Loading quarterback careers...")

    with open(
            datacube_filename,
            "r",
            encoding="utf-8",
            newline=""
    ) as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            if row:

                player_rows[
                    row[0]
                ].append(row)

    eligible_qbs = []

    for player, rows in player_rows.items():

        years = [
            int(row[YEAR_COLUMN])
            for row in rows
        ]

        if max(years) >= 2000:

            career_length = len(rows)

            eligible_qbs.append(
                (
                    player,
                    career_length
                )
            )

    eligible_qbs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    chosen_qbs = [
        player
        for player, career_length
        in eligible_qbs[:NUM_QBS]
    ]

    print()
    print(
        f"Selected {len(chosen_qbs)} quarterbacks:"
    )

    for player in chosen_qbs:

        print(player)

    player_careers = (
        get_all_player_careers(
            datacube_filename
        )
    )

    filtered_careers = {}

    for player in chosen_qbs:

        filtered_careers[player] = (
            player_careers[player]
        )

    player_names, career_matrix = (
        pad_careers_for_clustering(
            filtered_careers
        )
    )

    print()
    print("Building hierarchical clustering...")

    Z = linkage(career_matrix, method="average", metric="cosine")

    plt.figure(figsize=(20, 10))

    career_length_labels = [str(len(filtered_careers[player]))

        for player in player_names
    ]

    dendrogram(Z, labels=career_length_labels, leaf_rotation=90,
        leaf_font_size=10)

    plt.title("Hierarchical Clustering of Long-Career NFL Quarterbacks")
    plt.xlabel("Quarterback")
    plt.ylabel("Cluster Distance")
    plt.tight_layout()
    plt.savefig("QB_Dendrogram_Top40.png", dpi=300)
    plt.show()



main_dendrogram_recent_qbs()
