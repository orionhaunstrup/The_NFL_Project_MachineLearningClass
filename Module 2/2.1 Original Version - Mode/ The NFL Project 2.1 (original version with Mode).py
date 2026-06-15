""" The NFL Project - Machine Learning Class - Module 2

This version is the first one that performs the entire prediction set on
10000 experiments (with replacement) and then produced an evaluationn for
how accurate the predictions are!

It's the first version with a quantitative, numerical score for the
predictions.

Orion Haunstrup
Summer 2026
"""



from collections import defaultdict
import csv
import random
import numpy as np
import statistics
import random


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



def bin_player_percentiles(stats_percentiles, chosen_player_percentiles):
    ## This function is from my original version of the project
    ## where player predictions are done by BINNING the various
    ## predictions, generating a prediction for retirement time
    ## by when the MOST players by then have retired, and then
    ## getting the associated letter score from the pre-retirement
    ## years by which bin, A, B, C, D, or F, has the most votes.

    def assign_bin(val):
        if val is False:
            return 'Retired'
        if 0 <= val < 0.2:
            return 'F'
        if 0.2 <= val < 0.4:
            return 'D'
        if 0.4 <= val < 0.6:
            return 'C'
        if 0.6 <= val < 0.8:
            return 'B'
        if 0.8 <= val <= 1.0:
            return 'A'

    max_years = max(
        max(len(c) for c in stats_percentiles),
        len(chosen_player_percentiles)
    )

    def fluff(career):
        return career + [False] * (max_years - len(career))

    stats_fluffed = [
        fluff(career)
        for career in stats_percentiles
    ]

    chosen_fluffed = fluff(
        chosen_player_percentiles
    )

    # Count the bins for the neighbors
    the_stats_bins = []

    for year_idx in range(max_years):

        counts = {
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0,
            'F': 0,
            'Retired': 0
        }

        for career in stats_fluffed:

            bin_label = assign_bin(
                career[year_idx]
            )

            counts[bin_label] += 1

        the_stats_bins.append(counts)

    # Bin the chosen player
    chosen_player_bins = []

    for val in chosen_fluffed:

        chosen_player_bins.append(
            assign_bin(val)
        )

    return the_stats_bins, chosen_player_bins



def get_predicted_bins(
        the_stats_bins,
        years_revealed):

    predictions = []

    for year_idx in range(
            years_revealed,
            len(the_stats_bins)):

        counts = the_stats_bins[year_idx]

        max_count = max(counts.values())

        candidates = [
            b
            for b, count in counts.items()
            if count == max_count
        ]

        for b in PREFERRED_ORDER:

            if b in candidates:

                predictions.append(b)
                break

    return predictions



def convert_bins_to_percentiles(predicted_bins):
    ## This function is brand new to the project and lets me convert
    ## those predictive grade letters back into a numerical value

    predictions = []

    BIN_TO_PERCENTILE = {
    'A': 0.9,
    'B': 0.7,
    'C': 0.5,
    'D': 0.3,
    'F': 0.1,
    }

    for b in predicted_bins:

        if b == 'Retired':
            break  ## This makes it stop giving 'Retired' for each
                   ## season at the end of the vector.
                   ## ... Fixes a small glitch that was occurring.

        predictions.append(
            BIN_TO_PERCENTILE[b]
        )

    return predictions



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



def count_total_test_cases():

    total = 0

    for datacube_filename in DATACUBES.values():

        total += len(
            build_master_test_list(
                datacube_filename
            )
        )

    return total



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

        stats_percentiles = (
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

        the_stats_bins, chosen_player_bins = (
            bin_player_percentiles(
                stats_percentiles,
                chosen_player_percentiles
            )
        )

        predicted_bins = get_predicted_bins(
            the_stats_bins,
            years_revealed
        )

        predicted_percentiles = (
            convert_bins_to_percentiles(
                predicted_bins
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


main()
