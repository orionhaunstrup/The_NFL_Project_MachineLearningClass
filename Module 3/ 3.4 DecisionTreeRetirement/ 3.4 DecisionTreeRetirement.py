""" The NFL Project - Machine Learning Class - Module 3

This program builds the Decision Trees for retirement

Orion Haunstrup
Summer 2026
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


DATASETS = {
    "QB": "QB_DecisionTree_Data.csv",
    "Rushing": "Rushing_DecisionTree_Data.csv",
    "Receiving": "Receiving_DecisionTree_Data.csv",
    "Defense": "Defense_DecisionTree_Data.csv",
    "Kicking": "Kicking_DecisionTree_Data.csv"
}


def retirement_correction(retirement_difference):
    if retirement_difference <= -4:
        # We've predicted too early by 4+
        # We should add 4 years.
        return 4
    elif retirement_difference <= -2:
        # We've predicted too early by 2 or 3
        # We should dd 3 years.
        return 3
    elif retirement_difference <= 0:
        # We've predicted correctly or 1 year too early
        # We should add 1 year.
        return 1
    elif retirement_difference <= 2:
        # We've predicted too late by 1 or 2
        # We should subtract 1 year.
        return -1
    else:
        # We've predicted too late by 3+
        # We should subtract 3 years.
        return -3


def load_training_data(csv_filename):
    df = pd.read_csv(csv_filename)
    df["RETIREMENT_CORRECTION"] = (
        df["RETIREMENT_DIFFERENCE"].apply(retirement_correction))
    columns_to_drop = ["PLAYER",
                       "ACTUAL_REMAINING_SEASONS",
                       "PREDICTED_REMAINING_SEASONS",
                       "RETIREMENT_DIFFERENCE",
                       "RETIREMENT_CORRECTION",
                       "MEAN_PERCENTILE_DIFFERENCE"]
    if "RETIREMENT_BUCKET" in df.columns:
        columns_to_drop.append("RETIREMENT_BUCKET")
    X = df.drop(columns=columns_to_drop)
    y = df["RETIREMENT_CORRECTION"]
    feature_names = list(X.columns)
    return (X, y, feature_names)


def train_decision_tree(X, y):
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4, ## we'll only as far as 4, to not overfit
        random_state=333)
    tree.fit(X,y)
    return tree


def print_feature_importance(tree, feature_names):
    print()
    print("Feature Importances")
    print("-------------------")

    importances = zip(feature_names, tree.feature_importances_)
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    for feature, importance in importances:
        if importance > 0:
            print(
                f"{feature:30s}"
                f"{importance:.4f}"
            )


def evaluate_decision_tree(tree, X, y):
    predictions = tree.predict(X)
    accuracy = accuracy_score(y, predictions)
    print()
    print("Decision Tree Accuracy")
    print("----------------------")
    print(f"{accuracy:.2%}")


def split_training_data(X,y):
    return train_test_split(X, y, test_size=0.20,
                            random_state=333, stratify=y)


def print_retirement_tree(tree, feature_names, node=0, indent=""):
    left_child = (tree.tree_.children_left[node])
    right_child = (tree.tree_.children_right[node])
    
    # Leaf node
    if left_child == -1:
        prediction = tree.classes_[tree.tree_.value[node][0].argmax()]
        print(indent + f"Adjust Prediction: {prediction:+d}")
        return

    feature = (feature_names[tree.tree_.feature[node]])
    threshold = (tree.tree_.threshold[node])

    print(indent + f"{feature} <= {threshold:.2f}")
    print(indent + "├── Yes")
    print_retirement_tree(tree, feature_names, left_child, indent + "│   " )
    print(indent + "└── No")
    print_retirement_tree(tree, feature_names, right_child, indent + "    ")


class_names = [
    "Subtract 3 Years",
    "Subtract 1 Year",
    "Add 1 Year",
    "Add 3 Years",
    "Add 4 Years"
]


def draw_decision_tree(tree, feature_names, position):
    plt.figure(figsize=(27, 14))
    plot_tree(tree, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, impurity=True, fontsize=8)
    plt.title(f"{position} Retirement Decision Tree")
    plt.tight_layout()
    plt.savefig(f"{position}_DecisionTree.png", dpi=300)
    plt.close()


def main():

    for position, filename in DATASETS.items():

        print()
        print()
        print(position)

        X, y, feature_names = load_training_data(filename)
        (X_train, X_test, y_train, y_test) = split_training_data(X,y)
        tree = train_decision_tree(X_train, y_train)
        draw_decision_tree(tree, feature_names, position)
        print_feature_importance(tree, feature_names)
        evaluate_decision_tree(tree, X_test, y_test)
        print()
        print(f"{position} decision tree saved.")


main()
