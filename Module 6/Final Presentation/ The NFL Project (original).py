"""The NFL Project

This program, designed for my Data Mining class CSPB 4502,
lets you predict the future trajectory of an NFL player's career!

It does so in three different ways — Stats, HowGood_RawScore, Percentile.

Orion Haunstrup
Fall 2025
"""


### Changes to make (after turning in)
    ## Make retiring permanent, once engaged
    ## Make retiring 1/2 as strongly weighted, as a bin. Maybe even a 3rd
    ## Remove the original queried player from the training data
            ## (example Tom Brady at year 22 in the league)
    ## Build the testing bit. See what we can nudge to make this more
    ## and more accurate
    ## 


import random
import csv
from collections import defaultdict
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
        sus_players = [p for p in all_players if "#" in p]

        total_players = len(regular_players) + len(sus_players)

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
                filtered_sus = [
                    p for p in sus_players if p.upper().startswith(first_letter)]

                if filtered_regular or filtered_sus:
                    # Replace with narrowed lists
                    regular_players = filtered_regular
                    sus_players = filtered_sus
                    break   # exit loop because we succeeded

                print(f"No players found starting with '{first_letter}'. Try again!")

        # Print regular players
        if regular_players:
            print("\nPlease choose one of these players:")
            for player in regular_players:
                print(player)

        # Print sus players section header
        if sus_players:
            print()
            print("Or if you want to be a little spicy, this is a list of players")
            print("so new to the NFL that we only have ≤2 years of info on that")
            print("player. This program is not really designed to work with so")
            print("little info, but feel free to try if you'd like!")

            for player in sus_players:
                print(player.replace("#", "").strip())

        # Combine for validation
        valid_players = [p.replace('#','').strip() for p in regular_players + sus_players]

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



def bin_player_percentiles(stats_percentiles, howgood_percentiles, percentile_percentiles, 
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

    return the_stats_bins, the_howgood_bins, the_percentile_bins, the_chosen_player_bins


# Mapping of bin labels to y-values
BIN_TO_Y = {'Retired': -0.2, 'F': 0.1, 'D': 0.3, 'C': 0.5, 'B': 0.7, 'A': 0.9}
PREFERRED_ORDER = ['A','B','C','D','F','Retired']  # For tie-breaking

def plot_stats_percentiles(chosen_player_name, chosen_player_percentiles, stats_percentiles, the_stats_bins, num_comparison_years):
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
        plt.plot(range(1, max_years+1), career_plot, marker='o', color='skyblue', linewidth=1, markersize=4, alpha=0.7)
    
    # Chosen player
    chosen_plot = [y if y > -0.2 else np.nan for y in chosen_player_percentiles]
    plt.plot(range(1, max_years+1), chosen_plot, marker='o', color='black', linewidth=2, markersize=6)

    # Prediction start line
    plt.axvline(x=num_comparison_years, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.yticks([-0.2, 0.1, 0.3, 0.5, 0.7, 0.9], ['Retired', 'F', 'D', 'C', 'B', 'A'])
    plt.xlabel("Career Year")
    plt.ylabel("Percentile / Bin")
    plt.title(f"Chosen Player vs Closest Players (Stats Percentiles)")
    plt.ylim(-0.4, 1)
    
    # Legend with all elements
    purple_patch = Patch(facecolor='#9b59b6', alpha=0.5, label='Predicted most popular bin')
    black_line = Line2D([0], [0], color='black', linewidth=2, label=chosen_player_name)
    dotted_line = Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.7, label='Predictions start here')
    plt.legend(handles=[purple_patch, black_line, dotted_line], loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_howgood_percentiles(chosen_player_name, chosen_player_percentiles, howgood_percentiles, the_howgood_bins, num_comparison_years):
    plt.figure(figsize=(12, 6))
    max_years = len(chosen_player_percentiles)
    
    for year_idx in range(max_years):
        if year_idx >= num_comparison_years:
            counts = the_howgood_bins[year_idx]
            max_count = max(counts.values())
            candidates = [b for b, val in counts.items() if val == max_count]
            for b in PREFERRED_ORDER:
                if b in candidates:
                    most_popular_bin = b
                    break
            x_center = year_idx + 1
            y_center = BIN_TO_Y[most_popular_bin]
            rect = plt.Rectangle((x_center - 0.5, y_center - 0.1), 1.0, 0.2,
                                 facecolor='#9b59b6', alpha=0.5, edgecolor=None)
            plt.gca().add_patch(rect)
    
    for career in howgood_percentiles:
        career_plot = [y if y > -0.2 else np.nan for y in career]
        plt.plot(range(1, max_years+1), career_plot, marker='o', color='lightcoral', linewidth=1, markersize=4, alpha=0.7)
    
    chosen_plot = [y if y > -0.2 else np.nan for y in chosen_player_percentiles]
    plt.plot(range(1, max_years+1), chosen_plot, marker='o', color='black', linewidth=2, markersize=6)
    
    plt.axvline(x=num_comparison_years, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.yticks([-0.2, 0.1, 0.3, 0.5, 0.7, 0.9], ['Retired', 'F', 'D', 'C', 'B', 'A'])
    plt.xlabel("Career Year")
    plt.ylabel("Percentile / Bin")
    plt.title(f"Chosen Player vs Closest Players (HowGood Percentiles)")
    plt.ylim(-0.4, 1)
    
    purple_patch = Patch(facecolor='#9b59b6', alpha=0.5, label='Predicted most popular bin')
    black_line = Line2D([0], [0], color='black', linewidth=2, label=chosen_player_name)
    dotted_line = Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.7, label='Predictions start here')
    plt.legend(handles=[purple_patch, black_line, dotted_line], loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_percentile_percentiles(chosen_player_name, chosen_player_percentiles, percentile_percentiles, the_percentile_bins, num_comparison_years):
    plt.figure(figsize=(12, 6))
    max_years = len(chosen_player_percentiles)
    
    for year_idx in range(max_years):
        if year_idx >= num_comparison_years:
            counts = the_percentile_bins[year_idx]
            max_count = max(counts.values())
            candidates = [b for b, val in counts.items() if val == max_count]
            for b in PREFERRED_ORDER:
                if b in candidates:
                    most_popular_bin = b
                    break
            x_center = year_idx + 1
            y_center = BIN_TO_Y[most_popular_bin]
            rect = plt.Rectangle((x_center - 0.5, y_center - 0.1), 1.0, 0.2,
                                 facecolor='#9b59b6', alpha=0.5, edgecolor=None)
            plt.gca().add_patch(rect)
    
    for career in percentile_percentiles:
        career_plot = [y if y > -0.2 else np.nan for y in career]
        plt.plot(range(1, max_years+1), career_plot, marker='o', color='gray', linewidth=1, markersize=4, alpha=0.7)
    
    chosen_plot = [y if y > -0.2 else np.nan for y in chosen_player_percentiles]
    plt.plot(range(1, max_years+1), chosen_plot, marker='o', color='black', linewidth=2, markersize=6)
    
    plt.axvline(x=num_comparison_years, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.yticks([-0.2, 0.1, 0.3, 0.5, 0.7, 0.9], ['Retired', 'F', 'D', 'C', 'B', 'A'])
    plt.xlabel("Career Year")
    plt.ylabel("Percentile / Bin")
    plt.title(f"Chosen Player vs Closest Players (Percentile Careers)")
    plt.ylim(-0.4, 1)
    
    purple_patch = Patch(facecolor='#9b59b6', alpha=0.5, label='Predicted most popular bin')
    black_line = Line2D([0], [0], color='black', linewidth=2, label=chosen_player_name)
    dotted_line = Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.7, label='Predictions start here')
    plt.legend(handles=[purple_patch, black_line, dotted_line], loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

















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

    plot_stats_percentiles(chosen_player,
                           chosen_player_percentiles,
                           stats_percentiles,
                           the_stats_bins,
                           num_comparison_years)
    plot_howgood_percentiles(chosen_player,
                             chosen_player_percentiles,
                             howgood_percentiles,
                             the_howgood_bins,
                             num_comparison_years)
    plot_percentile_percentiles(chosen_player,
                                chosen_player_percentiles,
                                percentile_percentiles,
                                the_percentile_bins,
                                num_comparison_years)


    


def main():
    fortune_tell()

 
    
fortune_tell()
