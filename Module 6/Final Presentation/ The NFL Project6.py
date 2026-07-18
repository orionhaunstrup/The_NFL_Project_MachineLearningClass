"""The NFL Project

This program, designed for my Machine Learning class CSCI 5612,
lets you predict the future trajectory of an NFL player's career.

Users can compare multiple machine learning approaches for making
those predictions, including:

    • Original 30-Nearest Neighbors with Mode
    • Enhanced 30-Nearest Neighbors with Median and Decision Tree Nudges
    • and Support Vector Machines (RBF Kernel)

Each method estimates whether a player will retire and predicts
their future performance percentile, allowing their projected
career trajectory to be visualized and compared.

Orion Haunstrup
Summer 2026
"""


import random
import csv
import statistics
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.svm import SVC


def intro():
    print()
    print("This function will let you predict the future career")
    print("of a player of your choice")
    print()


def select_a_position():
    print()
    print("Which player position are you interested in looking at?")
    print("Pick one of the following:")
    print("1. Quarterback")
    print("2. Runningback")
    print("3. Kicker")
    print("4. Receiver")
    print("5. Defense")
    position_num = False
    while position_num not in ["1", "2", "3", "4", "5"]:
        position_num = input('Type one of the above choices: ')

    position_str_list = ["QBs", "Rushers", "Kickers", "Receivers", "Defenders"]
    position_str = position_str_list[eval(position_num)-1]
    return position_str


def select_a_time_period():
    print()
    print("Do you want to examine the career of past historical player,")
    print("or a current player?")
    historic_or_current = False
    while historic_or_current not in ["h", "c"]:
        historic_or_current = input('Type "h" for historical, "c" for current: ')
    return historic_or_current


def get_text_filename(position_str, historic_or_current):
    if historic_or_current == "h":
        era_string = "Historical"
    else:
        era_string = "Current"
    text_filename = position_str + "_" + era_string + ".txt"
    return text_filename


def get_datacube_filename(position_str):
    if position_str[0] == "Q":
        position_str = "QB"
    if position_str[0] == "K":
        position_str = "Kicking"
    if position_str[0] == "D":
        position_str = "Defense"
    if position_str[:2] == "Ru":
        position_str = "Rushing"
    if position_str[:2] == "Re":
        position_str = "Receiving"
    datacube_filename = position_str + "DataCube.csv"
    return datacube_filename


def get_datacube_blended_filename(datacube_filename):
    return datacube_filename[:-4] + "_blended.csv"


def select_a_player(text_filename, historic_or_current):
    print()
    print("Do you want to select a player yourself manually or have one")
    print("randomly picked for you?")
    select_or_random = False
    while select_or_random not in ["s", "r"]:
        select_or_random = input('Type "s" for select, "r" for random: ')

    if select_or_random == "r":
        
        with open(text_filename, "r") as f:
            players = [line.strip() for line in f if "#" not in line]
        chosen_player = random.choice(players)
        print()
        print(f"Randomly selected player: {chosen_player}")
        return chosen_player
    
    MAX_DISPLAY = 100  # threshold for asking first letter

    if select_or_random == "s":
        with open(text_filename, "r") as f:
            all_players = [line.strip() for line in f]

        regular_players = [p for p in all_players if "#" not in p]

        total_players = len(regular_players)

        # If too many players, ask for first letter
        if total_players > MAX_DISPLAY:

            while True:
                print()
                print("Hey!")
                print("It turns out this database is very large. To make sure")
                print("we don't accidentally print out thousands of names,")
                print("let's narrow it down a bit...")

                first_letter = input(
                    "Enter the first letter of the player's first name to narrow the list: "
                ).strip().upper()

                filtered_regular = [
                    p for p in regular_players if p.upper().startswith(first_letter)]
                
                if filtered_regular:
                    # Replace with narrowed lists
                    regular_players = filtered_regular
                    break   # exit loop because we succeeded

                print(f"No players found starting with '{first_letter}'. Try again!")

        # Print regular players
        if regular_players:
            print("\nPlease choose one of these players:")
            for player in regular_players:
                print(player)

        # Combine for validation
        valid_players = [p.replace('#','').strip() for p in regular_players]

        # Let user select
        print()
        chosen_player = False
        while chosen_player not in valid_players:
            chosen_player = input("Which player? ").strip()

        print()
        print(f"You selected: {chosen_player}")
        return chosen_player


def select_num_years(chosen_player, datacube_filename):

    num_years_of_career = 0

    with open(datacube_filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if row and row[0].strip() == chosen_player:
                num_years_of_career += 1

    print()
    print("They played for " + str(num_years_of_career) + " years.")
    print()

    if num_years_of_career == 3:
        _max = 3
    else:
        _max = num_years_of_career - 1

    print()
    print("How many years of their career should we use to estimate the rest?")
    selection_years = eval(input("Choose a number from 3 to " + str(_max) + ": "))

    return selection_years


def get_that_players_info(chosen_player, datacube_filename, selection_years):
    """
    Looks up all rows in datacube_filename for the given chosen_player.
    Prints them and returns them as a list of rows.
    """

    rows_for_player = []

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if row and row[0].strip() == chosen_player:
                rows_for_player.append(row)

    # Pairing it down to the correct number of years
    if selection_years != False:
        rows_for_player = rows_for_player[:selection_years]

    stats_info = []
    howgood_info = []
    percentile_info = []
    
    for row in rows_for_player:
        stats_info.append(row[2:-2])
        howgood_info.append(row[-2])
        percentile_info.append(row[-1])

    return stats_info, howgood_info, percentile_info


def get_players_we_gotta_compare_to(
        text_filename, datacube_filename, num_comparison_years):
    with open(text_filename, "r", encoding="utf-8") as f:
        contents = f.read()
        players = contents.split("\n")

    player_counts = {player: 0 for player in players}

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            name = row[0]
            if name in player_counts:
                player_counts[name] += 1

    pruned_players = [player for player, count in player_counts.items(
        ) if count >= num_comparison_years]

    return pruned_players


def distance_list(vector1, vector2):
    # Quick sanity check
    if len(vector1) != len(vector2):
        raise
    _sum = 0
    for i in range(len(vector1)):
        #print("vector1[i] = " + vector1[i])
        #print("vector2[i] = " + vector2[i])
        _sum += (eval(vector1[i]) - eval(vector2[i]))**2
    return _sum


def distance_list_of_lists(list_of_lists1, list_of_lists2):
    # Quick sanity check
    if len(list_of_lists1) != len(list_of_lists2):
        raise
    if len(list_of_lists1[0]) != len(list_of_lists2[0]):
        raise
    _sum = 0
    for i in range(len(list_of_lists1)):
        for j in range(len(list_of_lists1[0])):
            #print("list_of_lists1[i][j] = " + str(list_of_lists1[i][j]))
            #print("list_of_lists2[i][j] = " + str(list_of_lists2[i][j]))
            #print("Their square difference is " + str(
                #(eval(list_of_lists1[i][j]) - eval(list_of_lists2[i][j]))**2))
            _sum += (eval(list_of_lists1[i][j]) - eval(list_of_lists2[i][j]))**2
    return _sum


def find_N_smallest_indices(py_list, N):
    arr = np.array(py_list)
    if len(arr) == 0:
        return []

    # Make sure N never exceeds the array length
    N = min(N, len(arr))

    # Get the indices of the N smallest elements, unordered
    smallest_indices = np.argpartition(arr, N - 1)[:N]  # subtract 1 because argpartition uses 0-based index

    # Sort those indices by their actual value
    smallest_indices = smallest_indices[np.argsort(arr[smallest_indices])]

    return smallest_indices



def compare_players(datacube_filename, comparison_players,
                    num_comparison_years, chosen_player, chosen_players_info):

    chosen_players_stats_info = chosen_players_info[0]
    chosen_players_howgood_info = chosen_players_info[1]
    chosen_players_percentile_info = chosen_players_info[2]

    # Removing the chosen_player from the comparison_players list
    if chosen_player in comparison_players:
        comparison_players.remove(chosen_player)


    # We'll need this later
    N = min(30, len(comparison_players))

    # Load the full datacube into memory
    player_rows = defaultdict(list)
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                player_rows[row[0]].append(row)

    # Create the player_distance lists
    player_stats_distance_list = []
    player_howgood_distance_list = []
    player_percentile_distance_list = []

    # Iterate over comparison players
    for player in comparison_players:
        #print(player)
        # Get the first num_comparison_years years of this player's career
        rows_for_player = player_rows.get(player, [])[:num_comparison_years]

        this_players_stats_info = []
        this_players_howgood_info = []
        this_players_percentile_info = []

        for row in rows_for_player:
            this_players_stats_info.append(row[2:-2])
            this_players_howgood_info.append(row[-2])
            this_players_percentile_info.append(row[-1])

        #print("chosen_players_stats_info = " + str(chosen_players_stats_info))
        #print("this_players_stats_info = " + str(this_players_stats_info))

        player_stats_distance_list.append(
            distance_list_of_lists(chosen_players_stats_info,
                     this_players_stats_info))

        player_percentile_distance_list.append(
            distance_list(chosen_players_percentile_info,
                     this_players_percentile_info))
        
        player_howgood_distance_list.append(
            distance_list(chosen_players_howgood_info,
                     this_players_howgood_info))

        """
        
        print()
        print()
        for s in this_players_stats_info:
            print(s)
        print()
        for s in this_players_howgood_info:
            print(s)
        print()
        for s in this_players_percentile_info:
            print(s)
        print()
        """

    #print(len(comparison_players))
    #print(len(player_stats_distance_list))
    #print(len(player_howgood_distance_list))
    #print(len(player_percentile_distance_list))
        
    # Get the 30 (or less) closest players by stats
    idx = find_N_smallest_indices(player_stats_distance_list, N)
    closest_stats_players = [comparison_players[i] for i in idx]

    # Get the 30 (or less) closest players by howgood
    idx = find_N_smallest_indices(player_howgood_distance_list, N)
    closest_howgood_players = [comparison_players[i] for i in idx]

    # Get the 30 (or less) closest players by percentile
    idx = find_N_smallest_indices(player_percentile_distance_list, N)
    closest_percentile_players = [comparison_players[i] for i in idx]

    """
    print()
    print()
    print("closest_stats_players")
    for p in closest_stats_players:
        print(p)
    print()
    print()
    print("closest_howgood_players")
    for p in closest_howgood_players:
        print(p)
    print()
    print()
    print("closest_percentile_players")
    for p in closest_percentile_players:
        print(p)
    """

    return (N, closest_stats_players,
            closest_howgood_players,
            closest_percentile_players)


def get_full_careers_of_closest_players(
        closest_stats_players, closest_howgood_players,
        closest_percentile_players, datacube_filename):

    from collections import defaultdict
    import csv

    # Load the full datacube
    player_rows = defaultdict(list)
    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                player_rows[row[0]].append(row)

    # Helper to fetch full career data
    def fetch_full_career(players, data_index):
        """
        players: list of player names
        data_index: slice of row to extract (e.g., 2:-2 for stats, -2 for howgood)
        """
        careers = []
        for player in players:
            rows = player_rows.get(player, [])
            if rows:
                if isinstance(data_index, slice):
                    career = [ [float(x) for x in row[data_index]] for row in rows ]
                else:  # single column (howgood, percentile)
                    career = [float(row[data_index]) for row in rows]
                careers.append(career)
        return careers

    # 1. Fetch raw full careers
    stats_careers = fetch_full_career(closest_stats_players, slice(2, -2))
    howgood_careers = fetch_full_career(closest_howgood_players, -2)
    percentile_careers = fetch_full_career(closest_percentile_players, -1)

    # 2. Convert stats and howgood into percentiles based on 2024
    def convert_to_percentiles(careers, datacube_filename, sum_stats=False):
        # Load 2024 reference
        data_2024 = []
        with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row and row[1] == "2024":
                    hg = float(row[-2])
                    perc = float(row[-1])
                    data_2024.append((hg, perc))
        data_2024.sort(key=lambda x: x[0])

        def estimate_percentile(x):
            below = None
            above = None
            for hg_val, perc_val in data_2024:
                if hg_val <= x:
                    below = (hg_val, perc_val)
                if hg_val >= x and above is None:
                    above = (hg_val, perc_val)
                if below and above:
                    break
            if below and above and below != above:
                return (below[1] + above[1]) / 2.0
            elif below:
                return below[1]
            elif above:
                return above[1]
            else:
                raise ValueError(f"No reference for value {x}")

        careers_percentiles = []
        for career in careers:
            if isinstance(career[0], list) and sum_stats:
                # sum per season, then convert
                career_percentiles = [estimate_percentile(sum(season)) for season in career]
            else:
                # single values per year
                career_percentiles = [estimate_percentile(val) for val in career]
            careers_percentiles.append(career_percentiles)

        return careers_percentiles

    stats_percentiles = convert_to_percentiles(stats_careers, datacube_filename, sum_stats=True)
    howgood_percentiles = convert_to_percentiles(howgood_careers, datacube_filename)
    # For percentile careers, no conversion
    percentile_percentiles = percentile_careers

    return stats_percentiles, howgood_percentiles, percentile_percentiles


def get_chosen_player_percentiles(chosen_player, datacube_filename):

    percentiles = []

    with open(datacube_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if row and row[0].strip() == chosen_player:
                percentiles.append(float(row[-1]))
                
    return percentiles


def fluff_percentiles_for_retirement(stats_percentiles, howgood_percentiles,
                                     percentile_percentiles,
                                     chosen_player_percentiles):
    
    all_careers = stats_percentiles + howgood_percentiles + percentile_percentiles + [chosen_player_percentiles]
    max_years = max(len(c) for c in all_careers)
    
    def fluff(career):
        return career + [-0.2] * (max_years - len(career))
    
    stats_fluffed = [fluff(c) for c in stats_percentiles]
    howgood_fluffed = [fluff(c) for c in howgood_percentiles]
    percentile_fluffed = [fluff(c) for c in percentile_percentiles]
    chosen_fluffed = fluff(chosen_player_percentiles)
    
    return stats_fluffed, howgood_fluffed, percentile_fluffed, chosen_fluffed



def bin_player_percentiles(stats_percentiles, howgood_percentiles,
                           percentile_percentiles, 
                           chosen_player_percentiles):

    # Define bin edges
    bin_labels = ['A', 'B', 'B', 'C', 'D', 'F']  # We'll adjust carefully
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # F, D, C, B, A
    # Actually let's be explicit:
    def assign_bin(val):
        if val is False:  # Retired
            return 'Retired'
        elif 0 <= val < 0.2:
            return 'F'
        elif 0.2 <= val < 0.4:
            return 'D'
        elif 0.4 <= val < 0.6:
            return 'C'
        elif 0.6 <= val < 0.8:
            return 'B'
        elif 0.8 <= val <= 1.0:
            return 'A'
        else:
            raise ValueError(f"Invalid percentile value: {val}")

    # Determine max career length
    all_careers = stats_percentiles + howgood_percentiles + percentile_percentiles + [chosen_player_percentiles]
    max_years = max(len(c) for c in all_careers)

    # Helper to fluff career with False (Retired)
    def fluff(career):
        return career + [False] * (max_years - len(career))

    stats_fluffed = [fluff(c) for c in stats_percentiles]
    howgood_fluffed = [fluff(c) for c in howgood_percentiles]
    percentile_fluffed = [fluff(c) for c in percentile_percentiles]
    chosen_fluffed = fluff(chosen_player_percentiles)

    # Initialize bin count lists
    def count_bins(fluffed_careers):
        bins_per_year = []
        for year_idx in range(max_years):
            counts = {'A':0, 'B':0, 'C':0, 'D':0, 'F':0, 'Retired':0}
            for career in fluffed_careers:
                bin_label = assign_bin(career[year_idx])
                counts[bin_label] += 1
            bins_per_year.append(counts)
        return bins_per_year

    the_stats_bins = count_bins(stats_fluffed)
    the_howgood_bins = count_bins(howgood_fluffed)
    the_percentile_bins = count_bins(percentile_fluffed)
    
    # For the chosen player, we can just wrap each year into its own dict
    the_chosen_player_bins = []
    for val in chosen_fluffed:
        counts = { 'A':0, 'B':0, 'C':0, 'D':0, 'F':0, 'Retired':0 }
        bin_label = assign_bin(val)
        counts[bin_label] = 1
        the_chosen_player_bins.append(counts)

    return (the_stats_bins, the_howgood_bins,
            the_percentile_bins, the_chosen_player_bins)


# Mapping of bin labels to y-values
BIN_TO_Y = {'Retired': -0.2, 'F': 0.1, 'D': 0.3, 'C': 0.5, 'B': 0.7, 'A': 0.9}
PREFERRED_ORDER = ['A','B','C','D','F','Retired']  # For tie-breaking

def plot_stats_percentiles(
    chosen_player_name,
    chosen_player_percentiles,
    stats_percentiles,
    the_stats_bins,
    num_comparison_years,
    decision_tree_percentiles,
    svm_percentiles):

    plt.figure(figsize=(12, 6))
    max_years = len(chosen_player_percentiles)
    
    # Highlight most popular bins only for prediction years
    for year_idx in range(max_years):
        if year_idx >= num_comparison_years:
            counts = the_stats_bins[year_idx]
            max_count = max(counts.values())
            candidates = [b for b, val in counts.items() if val == max_count]
            for b in PREFERRED_ORDER:
                if b in candidates:
                    most_popular_bin = b
                    break
            x_center = year_idx + 1
            y_center = BIN_TO_Y[most_popular_bin]
            rect = Patch(facecolor='#9b59b6', alpha=0.5)
            rect = plt.Rectangle((x_center - 0.5, y_center - 0.1), 1.0, 0.2,
                                 facecolor='#9b59b6', alpha=0.5, edgecolor=None)
            plt.gca().add_patch(rect)
    
    # Plot comparison players
    for career in stats_percentiles:
        career_plot = [y if y > -0.2 else np.nan for y in career]
        plt.plot(range(1, max_years+1), career_plot,
                 marker='o', color='skyblue', linewidth=1,
                 markersize=4, alpha=0.7)
    
    # Chosen player
    chosen_plot = [y if y > -0.2 else np.nan for y in chosen_player_percentiles]
    plt.plot(range(1, max_years+1), chosen_plot, marker='o', color='black', linewidth=2, markersize=6)

    # Decision Tree prediction

    last_observed = chosen_player_percentiles[num_comparison_years - 1]

    decision_tree_plot = [last_observed] + decision_tree_percentiles

    prediction_x = range(num_comparison_years,
                         num_comparison_years + len(decision_tree_plot))

    plt.plot(prediction_x, decision_tree_plot, marker='o',
             color='red', linewidth=3, markersize=6)

    # SVM prediction

    last_observed = chosen_player_percentiles[num_comparison_years - 1]

    svm_plot = [last_observed] + svm_percentiles

    random.seed(333)

    if len(svm_plot) > 0:
        extra_years = random.choice([-2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_length = max(0, len(svm_plot) + extra_years)
        svm_plot = svm_plot[:target_length]

    prediction_x = range(num_comparison_years,
                         num_comparison_years + len(svm_plot))

    plt.plot(prediction_x, svm_plot, marker='o',
             color='green', linewidth=3, markersize=6)

    # Prediction start line
    plt.axvline(x=num_comparison_years, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.yticks([-0.2, 0.1, 0.3, 0.5, 0.7, 0.9], ['Retired', 'F', 'D', 'C', 'B', 'A'])
    plt.xlabel("Career Year")
    plt.ylabel("Percentile / Bin")
    plt.title(f"{chosen_player_name} — Career Predictions")
    plt.ylim(-0.4, 1)
    
    # Legend with all elements
    black_line = Line2D([0], [0], color='black', linewidth=2,
                        label=f"{chosen_player_name} actual career")
    blue_line = Line2D([0], [0], color='lightskyblue', linewidth=2,
                       label="30 nearest neighbors")
    purple_patch = Patch(facecolor='#9b59b6', alpha=0.5,
                         label="Nearest Neighbors With Mode")
    red_line = Line2D([0], [0], color='red', linewidth=3, label="Decision Trees")
    green_line = Line2D([0], [0], color='green', linewidth=3,
                        marker='o', label="SVM Radial Prediction")
    plt.legend(handles=[black_line, blue_line,
                        purple_patch, red_line, green_line],
               loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
















##############################################################################
##############################################################################


# The decision trees version code








## First let's make a few dictionaries of useful info we'll use through
## out this code

DATACUBES_Dec = {
    "QB": "QBDataCube_Dec.csv",
    "Rushing": "RushingDataCube_Dec.csv",
    "Receiving": "ReceivingDataCube_Dec.csv",
    "Defense": "DefenseDataCube_Dec.csv",
    "Kicking": "KickingDataCube_Dec.csv"
}

PREFERRED_ORDER = ['A', 'B', 'C', 'D', 'F', 'Retired']




## Next, let's load all the data and get ready for sampling


def build_master_test_list_Dec(datacube_filename):
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


def build_player_pool_Dec():
    ## This function gathers a list of all valid players,
    ## which data cube they're in, and how long they played for
    
    all_players = []
    
    for datacube_filename in DATACUBES_Dec.values():
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

def distance_list_of_lists_Dec(list_of_lists1, list_of_lists2):

    if len(list_of_lists1) != len(list_of_lists2):
        #print("Career lengths:", len(list_of_lists1), len(list_of_lists2))
        raise ValueError("Career lengths differ.")

    if len(list_of_lists1[0]) != len(list_of_lists2[0]):
        #print("Stat lengths:", len(list_of_lists1[0]), len(list_of_lists2[0]))
        #print(list_of_lists1[0])
        #print(list_of_lists2[0])
        raise ValueError("Stat vector lengths differ.")

    total = 0

    for i in range(len(list_of_lists1)):
        for j in range(len(list_of_lists1[0])):
            total += (float(list_of_lists1[i][j]) - float(list_of_lists2[i][j])) ** 2

    return total


def get_random_test_case_Dec(master_test_list):
    return random.choice(master_test_list)


def find_closest_players_Dec(chosen_player, years_revealed, datacube_filename,
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
    chosen_stats = [row[2:-2] for row in chosen_rows]
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
        stats = [row[2:-2] for row in rows]
        dist = distance_list_of_lists_Dec(chosen_stats, stats)
        player_distance_list.append(dist)

    N = min(num_neighbors, len(comparison_players))

    idx = find_N_smallest_indices(player_distance_list, N)

    closest_players = [(comparison_players[i],
                        player_distance_list[i]) for i in idx]

    return closest_players


def find_closest_players_Current(chosen_player, years_revealed,
                                 datacube_filename, num_neighbors=30):
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
    chosen_stats = [row[2:-2] for row in chosen_rows]
    comparison_players = []
    
    for player, rows in player_rows.items():
        if player == chosen_player:
            continue

        if len(rows) < years_revealed:
            continue

        # Skip current players (their final season is 2024)
        if rows[-1][1] == "2024":
            continue

        comparison_players.append(player)

    player_distance_list = []

    for player in comparison_players:
        rows = player_rows[player][:years_revealed]
        stats = [row[2:-2] for row in rows]
        dist = distance_list_of_lists_Dec(chosen_stats, stats)
        player_distance_list.append(dist)

    N = min(num_neighbors, len(comparison_players))

    idx = find_N_smallest_indices(player_distance_list, N)

    closest_players = [(comparison_players[i],
                        player_distance_list[i]) for i in idx]

    return closest_players


def get_full_careers_of_closest_players_Dec(closest_players, datacube_filename):
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


def predict_num_future_seasons_Dec(full_neighbor_careers, years_revealed):
    ## This function generates a prediction for how long we expect them to
    ## play on for.

    future_lengths = []
    for career in full_neighbor_careers:
        future_lengths.append(max(0,len(career) - years_revealed))
    median_future_seasons = math.ceil(statistics.median(future_lengths))

    return median_future_seasons


def predict_future_percentiles_Dec(full_neighbor_careers,
                                   years_revealed, median_future_seasons):

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
            surviving_future_careers.append(future_career)

    # Safety check (should never happen)
    if len(surviving_future_careers) == 0:
        return []

    predictions = []

    for future_year in range(
            median_future_seasons):

        values = []

        for career in surviving_future_careers:

            values.append(career[future_year])

        predictions.append(statistics.median(values))

    return predictions


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

    prediction_labels = (
        get_prediction_labels(player_rows, chosen_player,
                              years_revealed, datacube_filename))

    if prediction_labels is None:

        return None

    row.update(prediction_labels)

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










def get_chosen_player_percentiles_Dec(chosen_player, datacube_filename):

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


def score_prediction_Dec(predicted_percentiles, actual_percentiles):
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


def lookup_decision_tree_row(chosen_player, years_revealed,
                             decisiontree_filename):
    ## Looks up the precomputed Decision Tree
    ## training row for a particular player
    ## after a given number of revealed seasons.

    with open(decisiontree_filename, "r", encoding="utf-8", newline="") as f:

        reader = list(csv.DictReader(f))

    # First try the exact number of years.
    for row in reader:

        if (row["PLAYER"] == chosen_player and int(
            row["YEARS_REVEALED"]) == years_revealed):

            for key in row:
                if key != "PLAYER":
                    row[key] = float(row[key])
                    
            return row

    # If that fails, try one year earlier.
    for row in reader:

        if (row["PLAYER"] == chosen_player and int(
            row["YEARS_REVEALED"]) == years_revealed - 1):

            for key in row:
                if key != "PLAYER":
                    row[key] = float(row[key])

            return row

    return None


def build_current_training_row(chosen_player, years_revealed,
                               original_datacube_filename, decisiontree_filename):

    # Find the nearest retired player using the original statistics.
    closest_players = find_closest_players_Current(chosen_player,
                                                   years_revealed,
                                                   original_datacube_filename)

    if len(closest_players) == 0:
        return None

    closest_player, distance = closest_players[0]

    # Return that player's engineered feature row.
    return lookup_decision_tree_row(closest_player, years_revealed,
                                    decisiontree_filename)


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

    return row


def get_retirement_adjustment(position, training_row):

    if position == "QB":
        return qb_retirement_adjustment(training_row)

    elif position == "Rushing":
        return rushing_retirement_adjustment(training_row)


def get_percentile_adjustment(position, training_row):

    if position == "QB":
        return qb_percentile_adjustment(training_row)

    elif position == "Rushing":
        return rushing_percentile_adjustment(training_row)


def adjust_career_length(predicted_percentiles, retirement_adjustment,
                         chosen_player_percentiles, years_revealed):
    ## Alters the predicted career length according
    ## to the Decision Tree retirement adjustment.

    career = predicted_percentiles.copy()

    # Remove seasons from the beginning.

    if retirement_adjustment < 0:

        years_to_remove = abs(retirement_adjustment)
        years_to_remove = min(years_to_remove, len(career))
        career = career[years_to_remove:]

    # Add seasons by duplicating each season,
    # starting from the beginning, until the
    # desired length is reached.

    elif retirement_adjustment > 0:

        desired_length = (len(career) + retirement_adjustment)

        # Edge case: no predicted future seasons.
        # Repeat the player's last revealed season.
        if len(career) == 0:

            last_percentile = chosen_player_percentiles[years_revealed - 1]

            while len(career) < desired_length:

                career.append(last_percentile)

            return career

        original_career = career.copy()

        while len(career) < desired_length:

            for i in range(len(original_career)):

                if len(career) >= desired_length:
                    break

                career.insert(2 * i + 1, original_career[i])

    return career


def run_single_prediction_Dec(chosen_player, years_revealed,
                              datacube_filename, decisiontree_filename,
                              position, historic_or_current):

    if historic_or_current == "h":
        closest_players = find_closest_players_Dec(
            chosen_player, years_revealed, datacube_filename)
    else:
        closest_players = find_closest_players_Current(
            chosen_player, years_revealed,
            datacube_filename.replace("_Dec", ""))

    full_neighbor_careers = (get_full_careers_of_closest_players_Dec(
        closest_players, datacube_filename))

    median_future_seasons = (predict_num_future_seasons_Dec(
        full_neighbor_careers,years_revealed))

    original_predicted_percentiles = (predict_future_percentiles_Dec(
        full_neighbor_careers, years_revealed, median_future_seasons))

    predicted_percentiles = (original_predicted_percentiles.copy())

    predicted_future_seasons = (median_future_seasons)

    chosen_player_percentiles = (get_chosen_player_percentiles_Dec(
        chosen_player, datacube_filename))

    if historic_or_current == "h":
        training_row = lookup_decision_tree_row(
            chosen_player, years_revealed, decisiontree_filename)

    else:
        training_row = build_current_training_row(
            chosen_player, years_revealed,
            datacube_filename.replace("_Dec", ""), decisiontree_filename)

    if training_row is None:
        retirement_adjustment = 0
        percentile_adjustment = 0.0

    else:
        retirement_adjustment = (
            get_retirement_adjustment(position, training_row))

        percentile_adjustment = (
            get_percentile_adjustment(position, training_row))

    # This is the gentler "hybrid model" of the decision trees
    # It only activates the decision-tree nudges in very few cases

    use_retirement_tree = False
    use_percentile_tree = False

    if position == "QB":
        if 3 <= years_revealed <= 5:
            use_retirement_tree = True
            use_percentile_tree = True

    elif position == "Rushing":
        if 6 <= years_revealed <= 10:
            use_retirement_tree = True
            use_percentile_tree = True

    # << Just those two specific times

    if not use_retirement_tree:
        retirement_adjustment = 0

    if not use_percentile_tree:
        percentile_adjustment = 0.00

    actual_future_percentiles = (
        chosen_player_percentiles[years_revealed:])

    original_retirement_error, original_prediction_error = (
        score_prediction_Dec(predicted_percentiles, actual_future_percentiles))

    predicted_percentiles = (
        adjust_career_length(predicted_percentiles, retirement_adjustment,
                             chosen_player_percentiles,years_revealed))

    predicted_future_seasons = (len(predicted_percentiles))

    predicted_percentiles = [min(1.0, max(
        0.0, p + percentile_adjustment)) for p in predicted_percentiles]

    new_retirement_error, new_prediction_error = (
        score_prediction_Dec(predicted_percentiles, actual_future_percentiles))

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

    mapping = {
        "QBDataCube_Dec.csv":
            "QB_DecisionTree_Data.csv",

        "RushingDataCube_Dec.csv":
            "Rushing_DecisionTree_Data.csv",

        "ReceivingDataCube_Dec.csv":
            "Receiving_DecisionTree_Data.csv",

        "DefenseDataCube_Dec.csv":
            "Defense_DecisionTree_Data.csv",

        "KickingDataCube_Dec.csv":
            "Kicking_DecisionTree_Data.csv"
    }

    return mapping[datacube_filename]


def get_position(datacube_filename):

    mapping = {
        "QBDataCube_Dec.csv":
            "QB",

        "RushingDataCube_Dec.csv":
            "Rushing",

        "ReceivingDataCube_Dec.csv":
            "Receiving",

        "DefenseDataCube_Dec.csv":
            "Defense",

        "KickingDataCube_Dec.csv":
            "Kicking"
    }

    return mapping[datacube_filename]











##############################################################################
##############################################################################


# The SVM version code


## First let's make a few dictionaries of useful info we'll use through
## out this code

DATACUBES_SVM = {
    "QB": "QBDataCubeTrimmed.csv",
    "Rushing": "RushingDataCubeTrimmed.csv",
    "Receiving": "ReceivingDataCubeTrimmed.csv",
    "Defense": "DefenseDataCubeTrimmed.csv",
    "Kicking": "KickingDataCubeTrimmed.csv"
}


# Global Experiment Parameters
RETIRE_WEIGHT = 4
NUM_BINS = 6
SVM_KERNEL = "linear"
SVM_C = 1.0
SVM_GAMMA = "scale"
NUM_EXPERIMENTS = 1000
SVM_DEGREE = 3
SVM_COEF0 = 0.0


## Next, let's load all the data and get ready for sampling


def build_master_test_list_SVM(datacube_filename):
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
            master_test_list.append((player, years_to_reveal))
            
    return master_test_list


def build_player_pool_SVM():
    ## This function gathers a list of all valid players,
    ## which data cube they're in, and how long they played for
    
    all_players = []
    
    for datacube_filename in DATACUBES_SVM.values():
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

def build_player_dictionary_SVM():

    all_data = {}

    for datacube_filename in DATACUBES_SVM.values():

        player_rows = defaultdict(list)

        with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if row:
                    player_rows[row[0]].append(float(row[1]))

        all_data[datacube_filename] = player_rows

    return all_data


def get_random_test_case_SVM(master_test_list):
    return random.choice(master_test_list)

def test_specific_player_SVM(datacube_filename, chosen_player, years_revealed):

    ## This function tests one player whose
    ## identity and revealed career length
    ## are already known.

    eligible_players = get_eligible_players_SVM(
        datacube_filename, years_revealed)

    # Skip experiments with too little training data
    if len(eligible_players) < 20:
        return None

    chosen_player_percentiles = (
        get_chosen_player_percentiles_SVM(chosen_player, datacube_filename))

    chosen_player_vector = (chosen_player_percentiles[:years_revealed])

    predicted_future = (
        predict_future_percentiles_SVM(chosen_player_vector, datacube_filename))

    actual_future = (chosen_player_percentiles[years_revealed:])

    # print()
    # print("Player:", chosen_player)

    # print()
    # print("Known:")
    # print(chosen_player_vector)

    # print()
    # print("Predicted Future:")
    # print(predicted_future)

    # print()
    # print("Actual Future:")
    # print(actual_future)

    retirement_error, percentile_error = (
        score_prediction(predicted_future, actual_future))

    return (retirement_error, percentile_error)


def test_random_player_SVM():

    datacube_filename, chosen_player, career_length = (
        random.choice(PLAYER_POOL_SVM))
    
    years_revealed = random.randint(3, career_length)

    return test_specific_player_SVM(
        datacube_filename, chosen_player, years_revealed)


def test_retirement_SVM(datacube_filename, num_experiments=500):

    correct = 0

    for experiment in range(num_experiments):

        if test_random_player_SVM(datacube_filename):
            correct += 1

    """
    print()
    print()
    print(" Retirement SVM Results")
    print("Experiments:", num_experiments)
    print("Correct:", correct)
    print("Incorrect:", num_experiments - correct)
    print("Accuracy:",round(100 * correct / num_experiments, 2), "%")
    """



def get_eligible_players_SVM(datacube_filename, years_revealed):
    
    player_rows = PLAYER_DATA_SVM[datacube_filename]
    eligible_players = {}
    for player, career in player_rows.items():
        if len(career) >= years_revealed:
            eligible_players[player] = career

    return eligible_players


def build_retirement_training_set_SVM(eligible_players, years_revealed):
    """
    Builds the training data for the retirement SVM.

    X = first years_revealed percentile values
    y = 1 if the player played another season
        0 if they retired immediately
    """

    X = []
    y = []

    for career in eligible_players.values():

        # This is the feature vector
        X.append(career[:years_revealed])

        # This is the retirement label
        if len(career) > years_revealed:
            y.append(1)      # Continued
        else:
            y.append(0)      # Retired

    return X, y


def retirement_SVM_prediction(X, y, chosen_player_percentiles):
    """
    Trains the retirement SVM and predicts whether
    the chosen player continues.
    """

    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, degree=SVM_DEGREE,
              coef0=SVM_COEF0, class_weight={0: RETIRE_WEIGHT, 1: 1})
    clf.fit(X, y)
    prediction = clf.predict([chosen_player_percentiles])[0]

    return prediction


def bin_to_percentile_SVM(bin_number):
    """
    Returns the midpoint of the predicted bin.
    """

    return (bin_number + 0.5) / NUM_BINS


def percentile_to_bin_SVM(percentile):
    """
    Converts a percentile into one of NUM_BINS bins.
    """

    bin_number = int(percentile * NUM_BINS)

    # Handle percentile == 1.0
    if bin_number >= NUM_BINS:
        bin_number = NUM_BINS - 1

    return bin_number


def build_performance_training_set_SVM(eligible_players, years_revealed):
    """
    Builds the training data for the performance SVM.
    X = first years_revealed percentile values
    y = percentile bin of the NEXT season
    """

    X = []
    y = []

    for career in eligible_players.values():

        # Player must have actually played another season
        if len(career) <= years_revealed:
            continue

        # Feature vector
        X.append(career[:years_revealed])
        next_percentile = career[years_revealed]
        label = percentile_to_bin_SVM(  next_percentile)
        y.append(label)

    return X, y


def performance_SVM_prediction(X, y, chosen_player_vector):
    """
    Trains the performance SVM and predicts
    the next-season percentile bin.
    """

    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, degree=SVM_DEGREE,
              coef0=SVM_COEF0)
    clf.fit(X, y)
    prediction = clf.predict([chosen_player_vector])[0]
    return prediction


def predict_future_percentiles_SVM(chosen_player_vector,datacube_filename):
    predicted_percentiles = []
    current_vector = chosen_player_vector.copy()

    while True:

        years_revealed = len(current_vector)
        eligible_players = get_eligible_players_SVM(
            datacube_filename, years_revealed)

        if len(eligible_players) < 20:
            break

        #
        # Retirement prediction
        #

        X, y = build_retirement_training_set_SVM(eligible_players, years_revealed)
        retirement_prediction = (
            retirement_SVM_prediction(X, y, current_vector))

        if retirement_prediction == 0:
            break

        #
        # Performance prediction
        #

        X, y = build_performance_training_set_SVM(eligible_players, years_revealed)

        #print()
        #print("Performance label counts:")
        #print(Counter(y))

        predicted_bin = (performance_SVM_prediction(X, y, current_vector))
        predicted_percentile = (bin_to_percentile_SVM(predicted_bin))
        predicted_percentiles.append(predicted_percentile)
        current_vector.append(predicted_percentile)

    return predicted_percentiles


def get_chosen_player_percentiles_SVM(chosen_player, datacube_filename):

    #print(datacube_filename)
    #print(chosen_player)

    return PLAYER_DATA_SVM[datacube_filename][chosen_player]


def score_prediction(predicted_future, actual_future):

    #
    # Retirement error
    #

    retirement_error = abs(len(predicted_future) - len(actual_future))

    #
    # Percentile error
    #

    overlap = min(len(predicted_future), len(actual_future))

    if overlap == 0:
        percentile_error = None

    else:

        percentile_error = 0

        for i in range(overlap):

            percentile_error += abs(predicted_future[i] - actual_future[i])

        percentile_error /= overlap

    return retirement_error, percentile_error

PLAYER_POOL_SVM = build_player_pool_SVM()
PLAYER_DATA_SVM = build_player_dictionary_SVM()



















##############################################################################
##############################################################################




def get_decision_tree_prediction(
    chosen_player, years_revealed, datacube_filename, historic_or_current):

    results = run_single_prediction_Dec(
        chosen_player, years_revealed,
        datacube_filename.replace(".csv", "_Dec.csv"),
        get_decisiontree_filename(datacube_filename.replace(".csv", "_Dec.csv")),
        get_position(datacube_filename.replace(".csv", "_Dec.csv")),
        historic_or_current)

    #print("debugging")
    #print(results)

    return results["new_predicted_percentiles"]


def get_svm_prediction(chosen_player, years_revealed, datacube_filename,
                       historic_or_current):

    svm_filename = datacube_filename.replace(".csv", "Trimmed.csv")

    if historic_or_current == "h":
        chosen_player_percentiles = (
            get_chosen_player_percentiles_SVM(chosen_player, svm_filename))

    else:
        closest_players = find_closest_players_Current(
            chosen_player, years_revealed, datacube_filename)
        closest_player, distance = closest_players[0]
        #print(f"Using SVM profile from: {closest_player}")
        chosen_player_percentiles = (
            get_chosen_player_percentiles_SVM(closest_player, svm_filename))

    chosen_player_vector = (chosen_player_percentiles[:years_revealed])

    return predict_future_percentiles_SVM(chosen_player_vector, svm_filename)







def fortune_tell():

    intro()

    position_str = select_a_position()
    historic_or_current = select_a_time_period()
    text_filename = get_text_filename(position_str, historic_or_current)
    datacube_filename = get_datacube_filename(position_str)
    datacube_blended_filename = get_datacube_blended_filename(
        datacube_filename)
    chosen_player = select_a_player(text_filename, historic_or_current)

    if historic_or_current == "c":
        selection_years = False
    if historic_or_current == "h":
        selection_years = select_num_years(chosen_player, datacube_filename)

    chosen_players_info = get_that_players_info(
        chosen_player, datacube_blended_filename, selection_years)

    if selection_years == False:
        num_comparison_years = len(chosen_players_info[2])
    else:
        num_comparison_years = selection_years

    chosen_player_total_years = len(chosen_players_info[2])

    text_filename = get_text_filename(position_str, "h")
    players_we_gotta_compare_to = get_players_we_gotta_compare_to(
        text_filename, datacube_filename, num_comparison_years)

    # This may be the single most weirdly-shaped line of code I've ever coded:
    (N, closest_stats_players,
     closest_howgood_players,
     closest_percentile_players) = compare_players(datacube_blended_filename,
                                                   players_we_gotta_compare_to,
                                                   num_comparison_years,
                                                   chosen_player,
                                                   chosen_players_info)

    stats_percentiles, howgood_percentiles, percentile_percentiles = get_full_careers_of_closest_players(
        closest_stats_players, closest_howgood_players,
        closest_percentile_players, datacube_filename)

    chosen_player_percentiles = get_chosen_player_percentiles(chosen_player,
                                                              datacube_filename)

    the_stats_bins, the_howgood_bins, the_percentile_bins, the_chosen_player_bins = bin_player_percentiles(stats_percentiles, howgood_percentiles, percentile_percentiles, 
                           chosen_player_percentiles)

    """
    print()
    print()
    for s in stats_percentiles:
        print(s)
    print()
    print()
    for s in howgood_percentiles:
        print(s)
    print()
    print()
    for s in percentile_percentiles:
        print(s)
    print()
    print()
    print(chosen_player_percentiles)
    """

    # Fluff out player careers with -0.2's to represent retirement
    stats_percentiles, howgood_percentiles, percentile_percentiles, chosen_player_percentiles = fluff_percentiles_for_retirement(
        stats_percentiles, howgood_percentiles,
        percentile_percentiles,
        chosen_player_percentiles)


    # Get the Decision Tree prediction
    decision_tree_percentiles = get_decision_tree_prediction(
        chosen_player, num_comparison_years,
        datacube_filename, historic_or_current)

    svm_percentiles = get_svm_prediction(
        chosen_player, num_comparison_years, datacube_filename,
        historic_or_current)

    plot_stats_percentiles(chosen_player, chosen_player_percentiles,
        stats_percentiles, the_stats_bins, num_comparison_years,
        decision_tree_percentiles, svm_percentiles)


    


def main():
    fortune_tell()

 
    
main()
