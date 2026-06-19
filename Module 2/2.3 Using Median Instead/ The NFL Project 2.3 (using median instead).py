""" The NFL Project - Machine Learning Class - Module 2

This version will predict player's retirement year first and only
THEN move on to predicting their future how-good percentiles.

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


def get_random_test_case(master_test_list):
    return random.choice(master_test_list)


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


def find_closest_players(chosen_player, years_revealed, datacube_filename,
                         num_neighbors=30):
    ## This function finds and returns the 30 closest historical players,
    ## by their stats, and returns them. Note that it only examines
    ## players who played for at least as long as the given sample.

    player_rows = defaultdict(list)

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row:
                player_rows[row[0]].append(row)

    chosen_rows = player_rows[chosen_player][:years_revealed]
    chosen_stats = [
        row[2:-2]
        for row in chosen_rows
    ]
    comparison_players = []
    
    for player, rows in player_rows.items():
        if player == chosen_player:
            continue
        if len(rows) < years_revealed:
            continue
        comparison_players.append(player)

    player_distance_list = []

    for player in comparison_players:
        rows = player_rows[player][:years_revealed]
        stats = [
            row[2:-2]
            for row in rows
        ]
        dist = distance_list_of_lists(
            chosen_stats,
            stats
        )
        player_distance_list.append(dist)

    N = min(num_neighbors, len(comparison_players))

    idx = find_N_smallest_indices(
        player_distance_list,
        N
    )

    closest_players = [
        (
            comparison_players[i],
            player_distance_list[i]
        )
        for i in idx
    ]

    return closest_players


def get_full_careers_of_closest_players(closest_players, datacube_filename):
    ## This function goes and gets the full careers of those closest players

    player_rows = defaultdict(list)

    with open(datacube_filename, "r",
              encoding="utf-8",
              newline="") as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row:
                player_rows[row[0]].append(row)

    stats_percentiles = []

    for player, distance in closest_players:
        rows = player_rows.get(player, [])
        career_percentiles = [
            float(row[-1])     # The already-in-the-datacube percentile
            for row in rows
        ]
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


def predict_future_percentiles(
        full_neighbor_careers,
        years_revealed,
        median_future_seasons):

    # Special case:
    # We predict immediate retirement.
    if median_future_seasons == 0:
        return []

    # Keep only neighbors who lasted at least
    # median_future_seasons beyond years_revealed.
    surviving_future_careers = []

    for career in full_neighbor_careers:

        future_career = career[years_revealed:]

        if len(future_career) >= median_future_seasons:

            surviving_future_careers.append(
                future_career
            )

    # Safety check (should never happen)
    if len(surviving_future_careers) == 0:
        return []

    predictions = []

    for future_year in range(
            median_future_seasons):

        values = []

        for career in surviving_future_careers:

            values.append(
                career[future_year]
            )

        predictions.append(
            statistics.median(values)
        )

    return predictions


def test_random_player(datacube_filename):
    ## This function runs the whole experiment on one random player.
    ## Note that this does not actually run while inside of main()
    ## This exists strictly for testing and debugging.

    master_test_list = build_master_test_list(
        datacube_filename
    )

    print()
    print("Total test cases =", len(master_test_list))

    chosen_player, years_revealed = get_random_test_case(
        master_test_list
    )

    print()
    print("Chosen player:")
    print(chosen_player)

    print()
    print("Years revealed:")
    print(years_revealed)

    print()
    print("30 Closest Players")
    print("------------------")

    closest_players = find_closest_players(
        chosen_player,
        years_revealed,
        datacube_filename
    )

    for rank, (player, dist) in enumerate(
            closest_players,
            start=1):

        print(
            f"{rank:2d}. "
            f"{player:30s} "
            f"distance={dist:.2f}"
        )

    full_neighbor_careers = (
        get_full_careers_of_closest_players(
            closest_players,
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

    chosen_player_percentiles = (
        get_chosen_player_percentiles(
            chosen_player,
            datacube_filename
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

    print()
    print("Predicted Future Seasons:")
    print(median_future_seasons)

    print()
    print("Predicted Future Percentiles:")
    print(predicted_percentiles)

    print()
    print("Actual Future Percentiles:")
    print(actual_future_percentiles)

    print()
    print("Retirement Error:")
    print(retirement_error)

    print()
    print("Prediction Error:")
    print(prediction_error)


def get_chosen_player_percentiles(
        chosen_player,
        datacube_filename):

    percentiles = []

    with open(datacube_filename,
              "r",
              encoding="utf-8",
              newline="") as f:

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
        len(predicted_percentiles)
        - len(actual_percentiles)
    )

    overlap = min(
        len(predicted_percentiles),
        len(actual_percentiles)
    )

    if overlap == 0:

        prediction_error = None

    else:

        differences = []

        for i in range(overlap):

            differences.append(
                abs(
                    predicted_percentiles[i]
                    - actual_percentiles[i]
                )
            )

        prediction_error = (
            sum(differences)
            / len(differences)
        )

    return retirement_error, prediction_error



def main():
    ## Lastly, this function, of course, runs the full 10000 player experiment

    retirement_errors = []
    prediction_errors = []

    skipped_cases = 0

    # For reproducibility
    random.seed(333)

    NUM_EXPERIMENTS = 10000

    all_players = build_player_pool()

    print(
        f"Running {NUM_EXPERIMENTS:,} experiments..."
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

        closest_players = find_closest_players(
            chosen_player,
            years_revealed,
            datacube_filename
        )

        if len(closest_players) == 0:

            skipped_cases += 1
            continue

        full_neighbor_careers = (
            get_full_careers_of_closest_players(
                closest_players,
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

        if experiment_num % 100 == 0:

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
    print("Final Results:")

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


main()
