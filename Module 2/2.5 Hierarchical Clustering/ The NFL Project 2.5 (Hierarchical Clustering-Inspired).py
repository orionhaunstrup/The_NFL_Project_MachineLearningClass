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


def build_career_length_centroids(datacube_filename):
    ## Using Hierarchical Clustering as inspiration,
    ## this function clumps players together strictly
    ## based on their career lengths

    player_careers = (
        get_all_player_careers(
            datacube_filename
        )
    )

    careers_by_length = defaultdict(list)

    for player, career in player_careers.items():

        careers_by_length[
            len(career)
        ].append(career)

    centroids = {}

    for career_length, careers in (
            careers_by_length.items()):

        centroid = []

        for year in range(career_length):

            values = []

            for career in careers:

                values.append(
                    career[year]
                )

            centroid.append(
                statistics.mean(
                    values
                )
            )

        centroids[
            career_length
        ] = centroid

    return centroids


def find_closest_career_length(
        chosen_player_percentiles,
        years_revealed,
        centroids):
    ## This function takes the test player and finds
    ## which cluster centroid they're closest to

    best_length = None
    best_distance = float("inf")

    for career_length, centroid in (
            centroids.items()):

        if career_length < years_revealed:
            continue

        distance = 0

        for year in range(
                years_revealed):

            distance += (
                chosen_player_percentiles[
                    year
                ]
                - centroid[
                    year
                ]
            ) ** 2

        if distance < best_distance:

            best_distance = distance
            best_length = career_length

    return best_length



def predict_from_career_length_centroid(
        chosen_player_percentiles,
        years_revealed,
        centroids):
    ## This function then allows the code to generate the rest
    ## of the test player's career (as a prediction) based on
    ## the means of those career-length centroids

    predicted_length = (
        find_closest_career_length(
            chosen_player_percentiles,
            years_revealed,
            centroids
        )
    )

    predicted_percentiles = (
        centroids[predicted_length][
            years_revealed:
        ]
    )

    return (
        predicted_length,
        predicted_percentiles
    )



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



def main():

    random.seed(RANDOMSEED)

    NUM_EXPERIMENTS = 3000

    retirement_errors = []
    prediction_errors = []

    print("Building player pool...")

    all_players = build_player_pool()

    print("Building career-length centroids...")

    centroid_data = {}

    for datacube_filename in DATACUBES.values():

        centroid_data[
            datacube_filename
        ] = (
            build_career_length_centroids(
                datacube_filename
            )
        )

    print("Generating test cases...")
    print()

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

        chosen_player_percentiles = (
            get_chosen_player_percentiles(
                chosen_player,
                datacube_filename
            )
        )

        centroids = centroid_data[
            datacube_filename
        ]

        predicted_length = (
            find_closest_career_length(
                chosen_player_percentiles,
                years_revealed,
                centroids
            )
        )

        predicted_percentiles = (
            centroids[
                predicted_length
            ][years_revealed:]
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

        if experiment_num % 50 == 0:

            percent = (
                100.0
                * experiment_num
                / NUM_EXPERIMENTS
            )

            print(
                f"\rProgress: "
                f"{experiment_num:,}/"
                f"{NUM_EXPERIMENTS:,} "
                f"({percent:.1f}%)",
                end=""
            )

    print()
    print()
    print("FINAL RESULTS")
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
        len(
            prediction_errors
        )
    )


main()
