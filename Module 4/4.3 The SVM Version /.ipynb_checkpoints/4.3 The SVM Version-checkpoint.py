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
from sklearn.svm import SVC
from collections import Counter


## First let's make a few dictionaries of useful info we'll use through
## out this code

DATACUBES = {
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

def build_player_dictionary():
    all_data = {}

    for datacube_filename in DATACUBES.values():
        player_rows = defaultdict(list)
        with open(datacube_filename, "r", encoding="utf-8", newline="") as f:

            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if row:
                    player_rows[row[0]].append(float(row[1]))

        all_data[datacube_filename] = player_rows
        
    return all_data


def get_random_test_case(master_test_list):
    return random.choice(master_test_list)

def test_specific_player(datacube_filename, chosen_player, years_revealed):

    eligible_players = get_eligible_players(datacube_filename, years_revealed)

    # Skip experiments with too little training data
    if len(eligible_players) < 20:
        return None

    chosen_player_percentiles = (
        get_chosen_player_percentiles(chosen_player, datacube_filename))

    chosen_player_vector = (chosen_player_percentiles[:years_revealed])

    predicted_future = (
        predict_future_percentiles(chosen_player_vector, datacube_filename))

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

    return retirement_error, percentile_error


def test_random_player():

    datacube_filename, chosen_player, career_length = (
        random.choice(PLAYER_POOL))
    
    years_revealed = random.randint(3, career_length)

    return test_specific_player(
        datacube_filename, chosen_player, years_revealed)


def test_retirement_SVM(datacube_filename, num_experiments=500):

    correct = 0

    for experiment in range(num_experiments):

        if test_random_player(datacube_filename):
            correct += 1

    print()
    print()
    print(" Retirement SVM Results")
    print("Experiments:", num_experiments)
    print("Correct:", correct)
    print("Incorrect:", num_experiments - correct)
    print("Accuracy:",round(100 * correct / num_experiments, 2), "%")


def get_eligible_players(datacube_filename, years_revealed):

    player_rows = PLAYER_DATA[datacube_filename]

    eligible_players = {}

    for player, career in player_rows.items():

        if len(career) >= years_revealed:

            eligible_players[player] = career

    return eligible_players


def build_retirement_training_set(eligible_players, years_revealed):
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


def bin_to_percentile(bin_number):
    """
    Returns the midpoint of the predicted bin.
    """

    return (bin_number + 0.5) / NUM_BINS


def percentile_to_bin(percentile):
    """
    Converts a percentile into one of NUM_BINS bins.
    """

    bin_number = int(percentile * NUM_BINS)

    # Handle percentile == 1.0
    if bin_number >= NUM_BINS:
        bin_number = NUM_BINS - 1

    return bin_number


def build_performance_training_set(eligible_players, years_revealed):
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
        label = percentile_to_bin(  next_percentile)
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


def predict_future_percentiles(chosen_player_vector,datacube_filename):
    predicted_percentiles = []
    current_vector = chosen_player_vector.copy()

    while True:

        years_revealed = len(current_vector)
        eligible_players = get_eligible_players(
            datacube_filename, years_revealed)

        if len(eligible_players) < 20:
            break

        #
        # Retirement prediction
        #

        X, y = build_retirement_training_set(eligible_players, years_revealed)
        retirement_prediction = (retirement_SVM_prediction(X, y, current_vector))

        if retirement_prediction == 0:
            break

        #
        # Performance prediction
        #

        X, y = build_performance_training_set(eligible_players, years_revealed)

        #print()
        #print("Performance label counts:")
        #print(Counter(y))

        predicted_bin = (performance_SVM_prediction(X, y, current_vector))
        predicted_percentile = (bin_to_percentile(predicted_bin))
        predicted_percentiles.append(predicted_percentile)
        current_vector.append(predicted_percentile)

    return predicted_percentiles


def get_chosen_player_percentiles(chosen_player, datacube_filename):
    return PLAYER_DATA[datacube_filename][chosen_player]


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

    return retirement_error,percentile_error


def test_full_SVM(num_experiments=500):

    exact_matches = 0
    total_percentile_error = 0
    num_percentile_scores = 0
    total_retirement_error = 0
    skipped = 0
    

    for i in range(num_experiments):

        if (i + 1) % 25 == 0 or i == 0:
            print(f"Completed {i + 1} / {num_experiments}")

        result = test_random_player()

        if result is None:
            skipped += 1
            continue

        retirement_error, percentile_error = result
        total_retirement_error += retirement_error
        if percentile_error is not None:
            total_percentile_error += percentile_error
            num_percentile_scores += 1

        if retirement_error == 0:
            exact_matches += 1

    completed = num_experiments - skipped
    exact_accuracy = (100 * exact_matches / completed)
    average_retirement_error = (total_retirement_error / completed)
    if num_percentile_scores == 0:

        average_percentile_error = None

    else:

        average_percentile_error = (
            total_percentile_error / num_percentile_scores)

    return (exact_accuracy, average_retirement_error, average_percentile_error)


PLAYER_POOL = build_player_pool()
PLAYER_DATA = build_player_dictionary()

def main():

    global SVM_KERNEL
    global NUM_BINS
    global SVM_DEGREE

    experiments = [

        ("Linear", "linear", 6, 3),

        ("RBF", "rbf", 6, 3),

        ("Polynomial", "poly", 8, 4)

    ]

    for name, kernel, bins, degree in experiments:

        random.seed(333)

        print()
        print("=" * 50)
        print(name)
        print("=" * 50)

        print("Kernel:      ", kernel)
        print("Bins:        ", bins)
        print("Degree:      ", degree)
        print("Experiments: ", NUM_EXPERIMENTS)
        print()

        SVM_KERNEL = kernel
        NUM_BINS = bins
        SVM_DEGREE = degree

        accuracy, retirement_error, percentile_error = (
            test_full_SVM(NUM_EXPERIMENTS)
        )

        print()
        print("Results")
        print("-------")
        print("Kernel:", kernel)
        print("Bins:", bins)
        print("Degree:", degree)
        print("Mean Retirement Error:",
              round(retirement_error, 3))
        print("Mean Percentile Error:",
              round(percentile_error, 3))

main()
