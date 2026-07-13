""" The NFL Project - Machine Learning Class - Module 2

This version gathers the info we'll need for the decision trees.

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


def build_master_test_list(player_rows):
    master_test_list = []
    for player, rows in player_rows.items():
        career_length = len(rows)
        for years_to_reveal in range(3, career_length):
            master_test_list.append((player, years_to_reveal))
    return master_test_list



def load_player_careers(datacube_filename):
    player_rows = defaultdict(list)
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                player_rows[row[0]].append(row)
    return header, player_rows



def get_last_3_year_means(player_rows, header, chosen_player, years_revealed):
    """
    For every player/given_num_years pair, we need to obtain only
    the most recent 3 seasons of their info and get the means of
    those values. This will be the info entered into the top of
    the decision trees
    """

    career_rows = player_rows[chosen_player]
    revealed_rows = career_rows[:years_revealed]
    last_3_rows = revealed_rows[-3:]

    feature_dict = {}

    for col_idx in range(2, len(header)):
        values = []
        for row in last_3_rows:
            try:
                values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                pass
            
        if values:
            feature_dict[header[col_idx]] = statistics.mean(values)
            
    return feature_dict


def build_training_row(player_rows, header, chosen_player,
                       years_revealed, datacube_filename):

    row = {}
    row["PLAYER"] = chosen_player
    row["YEARS_REVEALED"] = years_revealed
    feature_dict = get_last_3_year_means(player_rows, header,
                                         chosen_player, years_revealed)
    row.update(feature_dict)
    row["ACTUAL_REMAINING_SEASONS"] = (get_actual_remaining_seasons(
        player_rows, chosen_player, years_revealed))
    prediction_labels = (get_prediction_labels(
        player_rows, chosen_player, years_revealed, datacube_filename))
    if prediction_labels is None:
        return None
    row.update(prediction_labels)
    return row


def get_actual_remaining_seasons(player_rows, chosen_player, years_revealed):
    career_length = len(player_rows[chosen_player])
    return (career_length - years_revealed)


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

    idx = find_N_smallest_indices(player_distance_list, N)

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

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

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


def predict_future_percentiles(full_neighbor_careers,
                               years_revealed, median_future_seasons):

    # In case of immediate retirement
    if median_future_seasons == 0:
        return []

    # Keep only neighbors who lasted at least
    # median_future_seasons beyond years_revealed.
    surviving_future_careers = []
    for career in full_neighbor_careers:
        future_career = career[years_revealed:]
        if len(future_career) >= median_future_seasons:
            surviving_future_careers.append(future_career)

    # A quick sanity check (should never happen)
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
    percentiles = []
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == chosen_player:
                percentiles.append(float(row[-1]))
    return percentiles


def get_prediction_labels(player_rows, chosen_player,
                          years_revealed,datacube_filename):
    
    closest_players = find_closest_players(
        chosen_player, years_revealed, datacube_filename)
    
    if len(closest_players) == 0:
        return None
    
    full_neighbor_careers = (
        get_full_careers_of_closest_players(
            closest_players, datacube_filename))

    predicted_remaining_seasons = (predict_num_future_seasons(
        full_neighbor_careers, years_revealed))

    predicted_percentiles = (predict_future_percentiles(
        full_neighbor_careers, years_revealed, predicted_remaining_seasons))

    actual_remaining_seasons = (get_actual_remaining_seasons(
        player_rows, chosen_player, years_revealed))

    actual_future_percentiles = (get_chosen_player_percentiles(
        chosen_player, datacube_filename)[years_revealed:])

    retirement_difference = (
        predicted_remaining_seasons - actual_remaining_seasons)

    percentile_differences = []

    for predicted, actual in zip(
        predicted_percentiles, actual_future_percentiles):
        percentile_differences.append(predicted - actual)

    if len(percentile_differences) > 0:
        mean_percentile_difference = (statistics.mean(
            percentile_differences))

    else:
        mean_percentile_difference = 0

    return {"PREDICTED_REMAINING_SEASONS":
            predicted_remaining_seasons,
            "RETIREMENT_DIFFERENCE":
            retirement_difference,
            "MEAN_PERCENTILE_DIFFERENCE":
            mean_percentile_difference}


def write_training_dataset(datacube_filename, output_filename):
    header, player_rows = load_player_careers(datacube_filename)
    master_test_list = build_master_test_list(player_rows)
    rows = []

    for player, years_revealed in master_test_list:
        row = build_training_row(player_rows, header, player,
                                 years_revealed, datacube_filename)
        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Rows written:", len(rows))



def write_position_dataset(position_name, datacube_filename):
    rows = []
    header, player_rows = (load_player_careers(datacube_filename))
    print()
    print(f"Building {position_name} dataset...")
    master_test_list = build_master_test_list(player_rows)
    total_cases = len(master_test_list)
    print(
        f"Total cases: "
        f"{total_cases:,}"
    )

    for case_num, (chosen_player, years_revealed) in enumerate(
        master_test_list, start=1):
        row = build_training_row(player_rows, header, chosen_player,
                                 years_revealed, datacube_filename)
        if row is None:
            continue

        rows.append(row)

        ## Adding a small progress bar, so I don't lose my mind
        if case_num % 100 == 0:
            percent = (100.0 * case_num / len(master_test_list))

            print(
                f"\r{position_name}: "
                f"{percent:.1f}%",
                end=""
            )

    output_filename = (
        f"{position_name}"
        f"_DecisionTree_Data.csv"
    )

    fieldnames = rows[0].keys()

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(
        f"Wrote {len(rows):,} rows to "
        f"{output_filename}"
    )






def main():

    total_positions = len(DATACUBES)

    for position_num, (position_name, datacube_filename) in enumerate(
            DATACUBES.items(),
            start=1):

        print()
        print(
            f"Position "
            f"{position_num}/"
            f"{total_positions}"
        )

        write_position_dataset(position_name, datacube_filename)

    print()
    print()
    print("All finished.")


main()
