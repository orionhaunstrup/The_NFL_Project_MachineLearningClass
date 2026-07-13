""" The NFL Project - Machine Learning Class - Module 2

This version combines the 30 closest players + median version
(the current high score holder) with the new Decision Trees-based
adjustments.

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
        for years_to_reveal in range(3, career_length):
            master_test_list.append((player, years_to_reveal))
            
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
            all_players.append(
                (datacube_filename, player, len(rows)))

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

            total += (float(
                list_of_lists1[i][j]) - float(list_of_lists2[i][j])) ** 2

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


def build_training_row(
        player_rows,
        header,
        chosen_player,
        years_revealed,
        datacube_filename):

    row = {}

    row["PLAYER"] = chosen_player
    row["YEARS_REVEALED"] = years_revealed

    feature_dict = get_last_3_year_means(
        player_rows,
        header,
        chosen_player,
        years_revealed
    )

    row.update(feature_dict)

    row["ACTUAL_REMAINING_SEASONS"] = (
        get_actual_remaining_seasons(
            player_rows,
            chosen_player,
            years_revealed
        )
    )

    prediction_labels = (
        get_prediction_labels(
            player_rows,
            chosen_player,
            years_revealed,
            datacube_filename
        )
    )

    if prediction_labels is None:

        return None

    row.update(
        prediction_labels
    )

    return row



"""
Now converting the Decision Trees into Python Code
"""

def qb_retirement_adjustment(training_row):

    if training_row["YEARS_REVEALED"] <= 11.50:

        if training_row["1ST DOWNS"] <= 1.39:

            if training_row["INTERCEPTIONS"] <= -1.21:

                if training_row["PASSING YARDS"] <= 1.22:
                    return -1
                else:
                    return 3

            else:

                if training_row["YEARS_REVEALED"] <= 3.50:
                    return -1
                else:
                    return 1

        else:

            if training_row["COMPLETION %"] <= 1.18:

                if training_row["COMPLETIONS"] <= 1.18:
                    return 4
                else:
                    return -3

            else:

                if training_row["FUMBLES"] <= -2.15:
                    return 3
                else:
                    return 4

    else:

        if training_row["YARDS PER ATTEMPT.1"] <= 0.11:

            if training_row["LONG"] <= 0.42:

                if training_row["FUMBLES"] <= -0.06:
                    return 1
                else:
                    return -1

            else:

                if training_row["LONG.1"] <= 0.15:
                    return -1
                else:
                    return 1

        else:

            if training_row["PERCENTILE"] <= 0.92:

                return 1

            else:

                if training_row["ATTEMPTS.1"] <= 0.23:
                    return 3
                else:
                    return -1



def rushing_retirement_adjustment(training_row):

    if training_row["YEARS_REVEALED"] <= 5.50:

        if training_row["HOWGOOD RAW SCORE"] <= 2.50:

            if training_row["PASS YARDS"] <= 0.00:

                if training_row["ATTEMPTS"] <= 0.03:
                    return 3
                else:
                    return 4

            else:

                return 1

        else:

            if training_row["YEARS_REVEALED"] <= 3.50:

                if training_row["PASS LONG"] <= 0.39:
                    return 1
                else:
                    return 3

            else:

                return 1

    else:

        if training_row["FUMBLES"] <= -2.54:

            if training_row["PERCENTILE"] <= 0.94:

                return 1

            else:

                if training_row["FUMBLES"] <= -3.63:
                    return -1
                else:
                    return 3

        else:

            return 1



def receiving_retirement_adjustment(training_row):

    if training_row["YEARS_REVEALED"] <= 5.50:

        if training_row["YARDS"] <= 0.23:

            return 1

        else:

            if training_row["YEARS_REVEALED"] <= 3.50:

                if training_row["RECEPTIONS"] <= 0.53:
                    return -1
                else:
                    return 1

            else:

                if training_row["PERCENTILE"] <= 0.50:
                    return -1
                else:
                    return 1

    else:

        if training_row["YARDS"] <= 0.47:

            return 1

        else:

            if training_row["YEARS_REVEALED"] <= 9.50:

                if training_row["40+"] <= 0.64:
                    return 1
                else:
                    return 4

            else:

                if training_row["RECEPTIONS"] <= 1.94:
                    return 1
                else:
                    return 3



def defense_retirement_adjustment(training_row):

    if training_row["YEARS_REVEALED"] <= 7.50:

        if training_row["YEARS_REVEALED"] <= 4.50:

            if training_row["HOWGOOD RAW SCORE"] <= 1.40:

                if training_row["TACKLE ASSISTS"] <= 0.02:
                    return -1
                else:
                    return 1

            else:

                if training_row["SACK LOSS YARDS CAUSED"] <= 0.00:
                    return -1
                else:
                    return 1

        else:

            return 1

    else:

        if training_row["TACKLE ASSISTS"] <= 0.16:

            if training_row["HOWGOOD RAW SCORE"] <= 0.00:

                if training_row["YEARS_REVEALED"] <= 14.50:
                    return 4
                else:
                    return 1

            else:

                return 1

        else:

            return 1



def kicking_retirement_adjustment(training_row):

    if training_row["YEARS_REVEALED"] <= 10.50:

        if training_row["FIELD GOALS MADE"] <= 1.21:

            if training_row["20-29"] <= 0.26:

                return -1

            else:

                if training_row["0-19"] <= 0.66:
                    return 1
                else:
                    return -3

        else:

            if training_row["40-49"] <= 0.90:

                return 1

            else:

                if training_row["HOWGOOD RAW SCORE"] <= 18.57:
                    return 4
                else:
                    return 1

    else:

        if training_row["40-49"] <= 1.60:

            if training_row["PERCENTILE"] <= 0.93:

                if training_row["EXTRA POINT RATIO"] <= 3.88:
                    return -1
                else:
                    return 4

            else:

                if training_row["POINTS"] <= 0.93:
                    return 4
                else:
                    return 3

        else:

            if training_row["HOWGOOD RAW SCORE"] <= 16.85:

                return -1

            else:

                if training_row["30-39"] <= 0.83:
                    return 3
                else:
                    return 1



def qb_percentile_adjustment(training_row):

    if training_row["PERCENTILE"] <= 0.93:

        if training_row["FUMBLES"] <= -1.35:

            if training_row["COMPLETION %"] <= 1.25:

                return -0.07

            else:

                if training_row["FUMBLES"] <= -1.97:
                    return 0.07
                else:
                    return -0.07

        else:

            if training_row["HOWGOOD RAW SCORE"] <= 6.20:

                return -0.07

            else:

                if training_row["SACK LOSS YARDS"] <= -0.66:
                    return -0.07
                else:
                    return 0.07

    else:

        if training_row["SACK LOSS YARDS"] <= -0.79:

            if training_row["TOUCHDOWNS.1"] <= 0.19:

                if training_row["YARDS PER ATTEMPT"] <= 1.05:
                    return -0.07
                else:
                    return 0.00

            else:

                if training_row["FUMBLES"] <= -2.34:
                    return -0.07
                else:
                    return 0.07

        else:

            if training_row["COMPLETION %"] <= 1.14:

                if training_row["20+"] <= 0.18:
                    return 0.07
                else:
                    return -0.07

            else:

                if training_row["TOUCHDOWNS.1"] <= 0.42:
                    return 0.07
                else:
                    return 0.00


def rushing_percentile_adjustment(training_row):

    if training_row["FUMBLES"] <= -1.14:

        if training_row["HOWGOOD RAW SCORE"] <= 7.51:

            if training_row["PERCENTILE"] <= 0.33:

                return -0.07

            else:

                if training_row["RUSHING 1ST %"] <= 1.34:
                    return -0.07
                else:
                    return 0.07

        else:

            if training_row["FUMBLES"] <= -3.63:

                if training_row["PASS YARDS PER RECEPTION"] <= 0.77:
                    return -0.07
                else:
                    return 0.00

            else:

                if training_row["YARDS PER ATTEMPT"] <= 1.70:
                    return 0.00
                else:
                    return 0.07

    else:

        if training_row["HOWGOOD RAW SCORE"] <= 7.53:

            if training_row["RUSHING 1ST DOWNS"] <= 0.07:

                return -0.07

            else:

                if training_row["FUMBLES"] <= -0.59:
                    return -0.07
                else:
                    return 0.07

        else:

            if training_row["YARDS PER ATTEMPT"] <= 1.65:

                if training_row["RUSHING 1ST DOWNS"] <= 0.76:
                    return 0.07
                else:
                    return -0.07

            else:

                return 0.07



def receiving_percentile_adjustment(training_row):

    if training_row["TOUCHDOWNS"] <= 0.43:

        if training_row["HOWGOOD RAW SCORE"] <= 2.63:

            if training_row["FUMBLES"] <= -0.81:

                if training_row["PERCENTILE"] <= 0.26:
                    return 0.00
                else:
                    return -0.07

            else:

                if training_row["40+"] <= 0.01:
                    return -0.07
                else:
                    return 0.07

        else:

            if training_row["RECEIVING 1ST DOWNS"] <= 0.40:

                if training_row["YEARS_REVEALED"] <= 4.50:
                    return -0.07
                else:
                    return 0.07

            else:

                return 0.00

    else:

        if training_row["RECEIVING 1ST DOWNS"] <= 0.88:

            return 0.00

        else:

            if training_row["40+"] <= 0.04:

                if training_row["TARGETS"] <= 0.02:
                    return 0.00
                else:
                    return 0.07

            else:

                return 0.00


def defense_percentile_adjustment(training_row):

    if training_row["SOLO TACKLES"] <= 0.00:

        if training_row["SACKS"] <= 0.01:

            if training_row["PERCENTILE"] <= 0.07:

                return -0.07

            else:

                if training_row["HOWGOOD RAW SCORE"] <= 0.67:
                    return 0.07
                else:
                    return -0.07

        else:

            if training_row["SACK LOSS YARDS CAUSED"] <= 0.01:

                return -0.07

            else:

                if training_row["YEARS_REVEALED"] <= 5.50:
                    return 0.07
                else:
                    return -0.07

    else:

        if training_row["PERCENTILE"] <= 0.91:

            if training_row["HOWGOOD RAW SCORE"] <= 1.29:

                if training_row["YEARS_REVEALED"] <= 6.50:
                    return 0.00
                else:
                    return 0.07

            else:

                if training_row["SACK LOSS YARDS CAUSED"] <= 0.00:
                    return -0.07
                else:
                    return 0.00

        else:

            if training_row["YEARS_REVEALED"] <= 11.50:

                return 0.00

            else:

                return 0.07


def kicking_percentile_adjustment(training_row):

    if training_row["PERCENTILE"] <= 0.81:

        if training_row["EXTRA POINT RATIO"] <= 1.08:

            return -0.07

        else:

            if training_row["YEARS_REVEALED"] <= 5.50:

                if training_row["40-49"] <= 0.39:
                    return 0.07
                else:
                    return -0.07

            else:

                return -0.07

    else:

        if training_row["FIELD GOALS MADE"] <= 1.08:

            if training_row["PERCENTILE"] <= 0.99:

                if training_row["POINTS"] <= 1.00:
                    return 0.07
                else:
                    return -0.07

            else:

                return 0.07

        else:

            if training_row["POINTS"] <= 1.30:

                return 0.07

            else:

                if training_row["PERCENTILE"] <= 0.97:
                    return -0.07
                else:
                    return 0.00

            










def test_random_player(
        datacube_filename,
        decisiontree_filename,
        position):
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

    original_predicted_percentiles = (
        predict_future_percentiles(
            full_neighbor_careers,
            years_revealed,
            median_future_seasons
        )
    )

    predicted_percentiles = (
        original_predicted_percentiles.copy()
    )

    predicted_future_seasons = (
        median_future_seasons
    )

    chosen_player_percentiles = (
        get_chosen_player_percentiles(
            chosen_player,
            datacube_filename
        )
    )

    training_row = (
        lookup_decision_tree_row(
            chosen_player,
            years_revealed,
            decisiontree_filename
        )
    )

    retirement_adjustment = (
        get_retirement_adjustment(
            position,
            training_row
        )
    )

    percentile_adjustment = (
        get_percentile_adjustment(
            position,
            training_row
        )
    )

    print()
    print("==============================")
    print("Decision Tree")
    print("==============================")

    print()
    print("Retirement Adjustment:")
    print(retirement_adjustment)

    print()
    print("Percentile Adjustment:")
    print(percentile_adjustment)

    actual_future_percentiles = (
        chosen_player_percentiles[
            years_revealed:
        ]
    )

    original_retirement_error, original_prediction_error = (
        score_prediction(
            predicted_percentiles,
            actual_future_percentiles
        )
    )

    print()
    print("==============================")
    print("Original Prediction")
    print("==============================")

    print()
    print("Predicted Future Seasons:")
    print(predicted_future_seasons)

    print()
    print("Predicted Future Percentiles:")
    print(predicted_percentiles)

    print()
    print("Actual Future Percentiles:")
    print(actual_future_percentiles)

    print()
    print("Original Retirement Error:")
    print(original_retirement_error)

    print()
    print("Original Prediction Error:")
    print(original_prediction_error)


    print()
    print("==============================")
    print("Applying Decision Tree Nudges")
    print("==============================")

    predicted_percentiles = (
        adjust_career_length(
            predicted_percentiles,
            retirement_adjustment,
            chosen_player_percentiles,
            years_revealed
        )
    )

    predicted_future_seasons = (
        len(predicted_percentiles)
    )

    predicted_percentiles = [
        min(
            1.0,
            max(
                0.0,
                p + percentile_adjustment
            )
        )
        for p in predicted_percentiles
    ]

    new_retirement_error, new_prediction_error = (
        score_prediction(
            predicted_percentiles,
            actual_future_percentiles
        )
    )

    print()
    print("==============================")
    print("Nudged Prediction")
    print("==============================")

    print()
    print("Predicted Future Seasons:")
    print(predicted_future_seasons)

    print()
    print("Predicted Future Percentiles:")
    print(predicted_percentiles)

    print()
    print("New Retirement Error:")
    print(new_retirement_error)

    print()
    print("New Prediction Error:")
    print(new_prediction_error)


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


def lookup_decision_tree_row(
        chosen_player,
        years_revealed,
        decisiontree_filename):
    ## Looks up the precomputed Decision Tree
    ## training row for a particular player
    ## after a given number of revealed seasons.

    with open(
            decisiontree_filename,
            "r",
            encoding="utf-8",
            newline="") as f:

        reader = list(csv.DictReader(f))

    # First try the exact number of years.
    for row in reader:

        if (
            row["PLAYER"] == chosen_player
            and
            int(row["YEARS_REVEALED"]) == years_revealed
        ):

            for key in row:

                if key != "PLAYER":

                    row[key] = float(row[key])

            return row

    # If that fails, try one year earlier.
    for row in reader:

        if (
            row["PLAYER"] == chosen_player
            and
            int(row["YEARS_REVEALED"]) == years_revealed - 1
        ):

            for key in row:

                if key != "PLAYER":

                    row[key] = float(row[key])

            return row

    return None


def get_retirement_adjustment(
        position,
        training_row):

    if position == "QB":
        return qb_retirement_adjustment(training_row)

    elif position == "Rushing":
        return rushing_retirement_adjustment(training_row)

    elif position == "Receiving":
        return receiving_retirement_adjustment(training_row)

    elif position == "Defense":
        return defense_retirement_adjustment(training_row)

    elif position == "Kicking":
        return kicking_retirement_adjustment(training_row)



def get_percentile_adjustment(
        position,
        training_row):

    if position == "QB":
        return qb_percentile_adjustment(training_row)

    elif position == "Rushing":
        return rushing_percentile_adjustment(training_row)

    elif position == "Receiving":
        return receiving_percentile_adjustment(training_row)

    elif position == "Defense":
        return defense_percentile_adjustment(training_row)

    elif position == "Kicking":
        return kicking_percentile_adjustment(training_row)



def adjust_career_length(
        predicted_percentiles,
        retirement_adjustment,
        chosen_player_percentiles,
        years_revealed):
    ## Alters the predicted career length according
    ## to the Decision Tree retirement adjustment.

    career = predicted_percentiles.copy()

    # -----------------------------------------
    # Remove seasons from the beginning.
    # -----------------------------------------

    if retirement_adjustment < 0:

        years_to_remove = abs(
            retirement_adjustment
        )

        years_to_remove = min(
            years_to_remove,
            len(career)
        )

        career = career[
            years_to_remove:
        ]

    # -----------------------------------------
    # Add seasons by duplicating each season,
    # starting from the beginning, until the
    # desired length is reached.
    # -----------------------------------------

    elif retirement_adjustment > 0:

        desired_length = (
            len(career)
            + retirement_adjustment
        )

        # Edge case: no predicted future seasons.
        # Repeat the player's last revealed season.
        if len(career) == 0:

            last_percentile = chosen_player_percentiles[
                years_revealed - 1
            ]

            while len(career) < desired_length:

                career.append(
                    last_percentile
                )

            return career

        original_career = career.copy()

        while len(career) < desired_length:

            for i in range(len(original_career)):

                if len(career) >= desired_length:
                    break

                career.insert(
                    2 * i + 1,
                    original_career[i]
                )

    return career


def run_single_prediction(
    chosen_player,
    years_revealed,
    datacube_filename,
    decisiontree_filename,
    position):

    closest_players = find_closest_players(
        chosen_player,
        years_revealed,
        datacube_filename
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

    original_predicted_percentiles = (
        predict_future_percentiles(
            full_neighbor_careers,
            years_revealed,
            median_future_seasons
        )
    )

    predicted_percentiles = (
        original_predicted_percentiles.copy()
    )

    predicted_future_seasons = (
        median_future_seasons
    )

    chosen_player_percentiles = (
        get_chosen_player_percentiles(
            chosen_player,
            datacube_filename
        )
    )

    training_row = (
        lookup_decision_tree_row(
            chosen_player,
            years_revealed,
            decisiontree_filename
        )
    )

    if training_row is None:

        return None

    retirement_adjustment = (
        get_retirement_adjustment(
            position,
            training_row
        )
    )

    percentile_adjustment = (
        get_percentile_adjustment(
            position,
            training_row
        )
    )

    actual_future_percentiles = (
        chosen_player_percentiles[
            years_revealed:
        ]
    )

    original_retirement_error, original_prediction_error = (
        score_prediction(
            predicted_percentiles,
            actual_future_percentiles
        )
    )

    predicted_percentiles = (
        adjust_career_length(
            predicted_percentiles,
            retirement_adjustment,
            chosen_player_percentiles,
            years_revealed
        )
    )

    predicted_future_seasons = (
        len(predicted_percentiles)
    )

    predicted_percentiles = [
        min(
            1.0,
            max(
                0.0,
                p + percentile_adjustment
            )
        )
        for p in predicted_percentiles
    ]

    new_retirement_error, new_prediction_error = (
        score_prediction(
            predicted_percentiles,
            actual_future_percentiles
        )
    )

    return {

        "chosen_player":
            chosen_player,

        "years_revealed":
            years_revealed,

        "retirement_adjustment":
            retirement_adjustment,

        "percentile_adjustment":
            percentile_adjustment,

        "original_retirement_error":
            original_retirement_error,

        "new_retirement_error":
            new_retirement_error,

        "original_prediction_error":
            original_prediction_error,

        "new_prediction_error":
            new_prediction_error,

        "original_predicted_percentiles":
            original_predicted_percentiles,

        "new_predicted_percentiles":
            predicted_percentiles,

        "actual_future_percentiles":
            actual_future_percentiles
    }


def get_decisiontree_filename(datacube_filename):
    ## Returns the corresponding Decision Tree data file.

    mapping = {
        "QBDataCube.csv":
            "QB_DecisionTree_Data.csv",

        "RushingDataCube.csv":
            "Rushing_DecisionTree_Data.csv",

        "ReceivingDataCube.csv":
            "Receiving_DecisionTree_Data.csv",

        "DefenseDataCube.csv":
            "Defense_DecisionTree_Data.csv",

        "KickingDataCube.csv":
            "Kicking_DecisionTree_Data.csv"
    }

    return mapping[datacube_filename]


def get_position(datacube_filename):
    ## Returns the position corresponding to the datacube.

    mapping = {
        "QBDataCube.csv":
            "QB",

        "RushingDataCube.csv":
            "Rushing",

        "ReceivingDataCube.csv":
            "Receiving",

        "DefenseDataCube.csv":
            "Defense",

        "KickingDataCube.csv":
            "Kicking"
    }

    return mapping[datacube_filename]



def main(
        datacube_filename,
        min_years_revealed,
        max_years_revealed):
    ## Lastly, this function, of course, runs the full 10000 player experiment

    # For reproducibility
    random.seed(777)

    NUM_EXPERIMENTS = 1000

    
    original_percentile_samples = 0
    new_percentile_samples = 0

    retirement_adjustment_counts = {
        -3: 0,
        -1: 0,
         1: 0,
         3: 0,
         4: 0
    }

    percentile_adjustment_counts = {
        -0.07: 0,
         0.00: 0,
         0.07: 0
    }

    original_retirement_error_sum = 0
    new_retirement_error_sum = 0

    original_percentile_error_sum = 0
    new_percentile_error_sum = 0

    retirement_improved = 0
    retirement_worsened = 0
    retirement_same = 0

    percentile_improved = 0
    percentile_worsened = 0
    percentile_same = 0

    master_test_list = build_master_test_list(datacube_filename)

    print(
        f"Running {NUM_EXPERIMENTS:,} experiments..."
    )

    for experiment_num in range(
            1,
            NUM_EXPERIMENTS + 1):


        eligible_players = [

            (player, years)

            for (
                player,
                years
            ) in master_test_list

            if (
                min_years_revealed
                <= years
                <= max_years_revealed
            )

        ]

        chosen_player, years_revealed = (
            random.choice(
                eligible_players
            )
        )

        decisiontree_filename = (
            get_decisiontree_filename(
                datacube_filename
            )
        )

        position = (
            get_position(
                datacube_filename
            )
        )

        results = run_single_prediction(
            chosen_player,
            years_revealed,
            datacube_filename,
            decisiontree_filename,
            position
        )

        if results is None:
            continue


        retirement_adjustment = (
            results["retirement_adjustment"]
        )

        percentile_adjustment = round(
            results["percentile_adjustment"],
            2
        )

        retirement_adjustment_counts[
            retirement_adjustment
        ] += 1

        percentile_adjustment_counts[
            percentile_adjustment
        ] += 1

        original_retirement_error = (
            results["original_retirement_error"]
        )

        new_retirement_error = (
            results["new_retirement_error"]
        )

        original_percentile_error = (
            results["original_prediction_error"]
        )

        new_percentile_error = (
            results["new_prediction_error"]
        )

        original_retirement_error_sum += (
            original_retirement_error
        )

        new_retirement_error_sum += (
            new_retirement_error
        )

        if original_percentile_error is not None:

            original_percentile_error_sum += (
                original_percentile_error
            )

            original_percentile_samples += 1


        if new_percentile_error is not None:

            new_percentile_error_sum += (
                new_percentile_error
            )

            new_percentile_samples += 1

        if retirement_adjustment != 0:

            if (
                new_retirement_error
                < original_retirement_error
            ):

                retirement_improved += 1

            elif (
                new_retirement_error
                > original_retirement_error
            ):

                retirement_worsened += 1

            else:

                retirement_same += 1

        if (
            percentile_adjustment != 0
            and
            original_percentile_error is not None
            and
            new_percentile_error is not None
        ):

            if (
                new_percentile_error
                < original_percentile_error
            ):

                percentile_improved += 1

            elif (
                new_percentile_error
                > original_percentile_error
            ):

                percentile_worsened += 1

            else:

                percentile_same += 1


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
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print("Retirement Adjustment Distribution")
    print("---------------------------------")

    for adjustment in [-3, -1, 1, 3, 4]:

        percent = (
            100
            * retirement_adjustment_counts[adjustment]
            / NUM_EXPERIMENTS
        )

        print(
            f"{adjustment:+2d} Years : "
            f"{percent:6.2f}%"
        )

    print()
    print("Percentile Adjustment Distribution")
    print("----------------------------------")

    for adjustment in [-0.07, 0.00, 0.07]:

        percent = (
            100
            * percentile_adjustment_counts[adjustment]
            / NUM_EXPERIMENTS
        )

        print(
            f"{adjustment:+.2f} : "
            f"{percent:6.2f}%"
        )
        
    print()
    print("Mean Retirement Error")
    print("---------------------")

    print(
        "Original:",
        original_retirement_error_sum
        / NUM_EXPERIMENTS
    )

    print(
        "Nudged:  ",
        new_retirement_error_sum
        / NUM_EXPERIMENTS
    )

    print()
    print("Mean Percentile Error")
    print("---------------------")

    print(
        "Original:",
        original_percentile_error_sum
        / original_percentile_samples
    )

    print(
        "Nudged:  ",
        new_percentile_error_sum
        / new_percentile_samples
    )

    retirement_total = (
        retirement_improved
        + retirement_worsened
        + retirement_same
    )

    percentile_total = (
        percentile_improved
        + percentile_worsened
        + percentile_same
    )

    print()
    print("Retirement Nudges")
    print("-----------------")

    if retirement_total > 0:

        print(
            f"Improved : "
            f"{100 * retirement_improved / retirement_total:.2f}%"
        )

        print(
            f"Worsened : "
            f"{100 * retirement_worsened / retirement_total:.2f}%"
        )

        print(
            f"Same      : "
            f"{100 * retirement_same / retirement_total:.2f}%"
        )

    print()
    print("Percentile Nudges")
    print("-----------------")

    if percentile_total > 0:

        print(
            f"Improved : "
            f"{100 * percentile_improved / percentile_total:.2f}%"
        )

        print(
            f"Worsened : "
            f"{100 * percentile_worsened / percentile_total:.2f}%"
        )

        print(
            f"Same      : "
            f"{100 * percentile_same / percentile_total:.2f}%"
        )

print()
print("QBs 3-5")
main(
    "QBDataCube.csv",
    3,
    5
)

print()
print("QBs 6-10")
main(
    "QBDataCube.csv",
    6,
    10
)

print()
print("Rushers 3-5")
main(
    "RushingDataCube.csv",
    3,
    5
)

print()
print("Rushers 6-10")
main(
    "RushingDataCube.csv",
    6,
    10
)

print()
print("Receiving 3-5")
main(
    "ReceivingDataCube.csv",
    3,
    5
)

print()
print("Receiving 6-10")
main(
    "ReceivingDataCube.csv",
    6,
    10
)

print()
print("Defense 3-5")
main(
    "DefenseDataCube.csv",
    3,
    5
)

print()
print("Defense 6-10")
main(
    "DefenseDataCube.csv",
    6,
    10
)

print()
print("Kicking 3-5")
main(
    "KickingDataCube.csv",
    3,
    5
)

print()
print("Kicking 6-10")
main(
    "KickingDataCube.csv",
    6,
    10
)
