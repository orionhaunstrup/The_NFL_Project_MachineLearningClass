""" The NFL Project - Machine Learning Class - Module 2

This version finally gets K-Means up and running by first
solving The Jagged Cube Problem

Orion Haunstrup
Summer 2026
"""



from collections import defaultdict
import csv
import random
import numpy as np
import statistics
import random
import math
from sklearn.decomposition import PCA


## First let's make a few dictionaries of useful info we'll use through
## out this code
DATACUBES = {
    "QB": "QBDataCube.csv",
    "Rushing": "RushingDataCube.csv",
    "Receiving": "ReceivingDataCube.csv",
    "Defense": "DefenseDataCube.csv",
    "Kicking": "KickingDataCube.csv"
}



## This is a new global variable, designed to store all
## the transformed data cubes in local memory
global PRECOMPUTED
PRECOMPUTED = {}

def build_precomputed_cubes():
    # This makes the datacube transformations we'll store in local memory

    for datacube_filename in DATACUBES.values():
        player_rows = defaultdict(list)
        with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    player_rows[row[0]].append(row)
        for years_revealed in range(3, 11):
            player_names = []
            player_vectors = []
            for player, rows in player_rows.items():
                if len(rows) < years_revealed:
                    continue
                truncated_rows = rows[:years_revealed]
                vector = []
                for row in truncated_rows:
                    # Remove the unnessary columns
                    # PLAYER NAME
                    # YEAR
                    # HOWGOOD RAW SCORE
                    # PERCENTILE
                    vector.extend(row[2:-2])
                player_names.append(player)
                player_vectors.append([float(x) for x in vector])
            PRECOMPUTED[(datacube_filename, years_revealed)] = {
                "player_names": player_names,
                "vectors": np.array(player_vectors, dtype=float)}
    print("Precomputed cubes built.")



def find_pca_neighbors(
        chosen_player,
        years_revealed,
        datacube_filename,
        num_neighbors=30):

    ## This function performs PCA on the non-jagged
    ## precomputed data cube and then finds the
    ## nearest neighbors in PCA-space.

    data = PRECOMPUTED[
        (datacube_filename,
         years_revealed)
    ]

    player_names = data["player_names"]
    vectors = data["vectors"]

    # Locate and remove the test player

    chosen_index = player_names.index(
        chosen_player
    )

    test_vector = vectors[
        chosen_index
    ]

    training_names = (
        player_names[:chosen_index]
        +
        player_names[chosen_index + 1:]
    )

    training_vectors = np.delete(
        vectors,
        chosen_index,
        axis=0
    )

    # Perform PCA

    pca = PCA(
        n_components=PCA_VARIANCE
    )

    pca.fit(
        training_vectors
    )

    training_pca = pca.transform(
        training_vectors
    )

    test_pca = pca.transform(
        test_vector.reshape(1, -1)
    )[0]

    # Compute distances

    player_distance_list = []

    for i, player_vector in enumerate(
            training_pca):

        distance = np.linalg.norm(
            player_vector
            - test_pca
        )

        player_distance_list.append(
            (
                training_names[i],
                distance
            )
        )

    player_distance_list.sort(
        key=lambda x: x[1]
    )

    return player_distance_list[
        :num_neighbors
    ]




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
        with open(datacube_filename,
                  "r",
                  encoding="utf-8",
                  newline="") as f:
            reader = csv.reader(f)
            next(reader)
            
            for row in reader:
                if row:
                    player_rows[row[0]].append(row)
                    
        for player, rows in player_rows.items():
            all_players.append(
                (
                    datacube_filename,
                    player,
                    len(rows)
                )
            )

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


def get_full_careers_of_cluster_players(cluster_players, datacube_filename):
    ## This function goes and gathers all the info from the original
    ## before-being-processed datacubes on the cluster players

    player_rows = defaultdict(list)
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row:
                player_rows[row[0]].append(row)

    stats_percentiles = []

    for player in cluster_players:

        rows = player_rows.get(player, [])

        career_percentiles = [
            float(row[-1])
            for row in rows]

        stats_percentiles.append(career_percentiles)

    return stats_percentiles


def predict_num_future_seasons(full_neighbor_careers, years_revealed):
    ## This function generates a prediction for how long we expect them to
    ## play on for.

    future_lengths = []

    for career in full_neighbor_careers:

        future_lengths.append(max(0,len(career) - years_revealed))

    median_future_seasons = math.ceil(statistics.median(future_lengths))

    return median_future_seasons


def predict_future_percentiles(full_neighbor_careers, years_revealed,
                               median_future_seasons):

    # This is a special case. If the prediction is that
    # They'll play 0 future seasons, then no prediction
    # percentiles for their future seasons are necessary
    if median_future_seasons == 0:
        return []

    # Keep only neighbors who lasted at least
    # median_future_seasons beyond years_revealed.
    surviving_future_careers = []

    for career in full_neighbor_careers:

        future_career = career[years_revealed:]

        if len(future_career) >= median_future_seasons:

            surviving_future_careers.append(future_career)

    # Quick sanity check (this should never happen)
    if len(surviving_future_careers) == 0:
        return []

    predictions = []

    for future_year in range(median_future_seasons):
        values = []
        for career in surviving_future_careers:
            values.append(career[future_year])

        predictions.append(statistics.median(values))

    return predictions


def get_chosen_player_percentiles(chosen_player, datacube_filename):
    ## To compare the predictions to the actual 'real' career that
    ## test player played in the NFL, we need to go get that player's
    ## historical data.

    percentiles = []

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            if row[0] == chosen_player:
                percentiles.append(float(row[-1]))
                    # The already-in-the-datacube percentile

    return percentiles


def score_prediction(predicted_percentiles, actual_percentiles):
    ## This function is brand new to the projec and lets me compare
    ## the machine's prediction to the actual rest of that player's
    ## career and see how accurate the prediction was!

    retirement_error = abs(len(predicted_percentiles) - len(actual_percentiles))
    overlap = min(len(predicted_percentiles), len(actual_percentiles))

    if overlap == 0:
        prediction_error = None

    else:
        differences = []
        for i in range(overlap):
            differences.append(abs(
                predicted_percentiles[i] - actual_percentiles[i]))
        prediction_error = (sum(differences) / len(differences))

    return retirement_error, prediction_error



def main():

    build_precomputed_cubes()

    retirement_errors = []
    prediction_errors = []

    skipped_cases = 0

    random.seed(333)

    NUM_EXPERIMENTS = 10000

    all_players = build_player_pool()

    print(
        f"Running {NUM_EXPERIMENTS:,} PCA experiments..."
    )

    for experiment_num in range(
            1,
            NUM_EXPERIMENTS + 1):

        years_revealed = random.randint(
            3,
            10
        )

        eligible_players = [

            (
                datacube_filename,
                player
            )

            for (
                datacube_filename,
                player,
                career_length
            ) in all_players

            if career_length >= years_revealed

        ]

        datacube_filename, chosen_player = (
            random.choice(
                eligible_players
            )
        )

        closest_players = (
            find_pca_neighbors(
                chosen_player,
                years_revealed,
                datacube_filename
            )
        )

        if len(closest_players) == 0:

            skipped_cases += 1
            continue

        full_neighbor_careers = (
            get_full_careers_of_cluster_players(
                [
                    player
                    for player, distance
                    in closest_players
                ],
                datacube_filename
            )
        )

        chosen_player_percentiles = (
            get_chosen_player_percentiles(
                chosen_player,
                datacube_filename
            )
        )

        median_future_seasons = (
            predict_num_future_seasons(
                full_neighbor_careers,
                years_revealed
            )
        )

        predicted_percentiles = (
            predict_future_percentiles(
                full_neighbor_careers,
                years_revealed,
                median_future_seasons
            )
        )

        actual_future_percentiles = (
            chosen_player_percentiles[
                years_revealed:
            ]
        )

        retirement_error, prediction_error = (
            score_prediction(
                predicted_percentiles,
                actual_future_percentiles
            )
        )

        retirement_errors.append(
            retirement_error
        )

        if prediction_error is not None:

            prediction_errors.append(
                prediction_error
            )

        if experiment_num % 200 == 0:

            percent = (
                100.0
                * experiment_num
                / NUM_EXPERIMENTS
            )

            print(percent, end="")

    print()
    print()

    print("Final Results:")
    print()

    print(
        "Mean Retirement Error:",
        statistics.mean(
            retirement_errors
        )
    )

    print()

    print(
        "Mean Prediction Error:",
        statistics.mean(
            prediction_errors
        )
    )

    print()

    print(
        "Prediction Error Samples:",
        len(prediction_errors)
    )

    print(
        "Skipped Cases:",
        skipped_cases
    )



global PCA_VARIANCE

PCA_VARIANCE = 0.85
main()
