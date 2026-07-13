"""
Decision Tree Demo
Orion Haunstrup
Summer 2026
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = "Demo_Data.csv"

TARGET = "PREDICTION CATEGORY"


# Load the data

def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["PLAYER", "RETIREMENT_DIFFERENCE", TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    return X, y, feature_names


# Train the little demo tree

def train_tree(X, y):
    tree = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                  random_state=333) # use the usual random seed
    tree.fit(X, y)
    return tree


# Print the feature importances

def print_feature_importance(tree, feature_names):

    print()
    print()
    print("FEATURE IMPORTANCES")
    print()

    importances = sorted(
        zip(feature_names, tree.feature_importances_),
        key=lambda x: x[1], reverse=True)

    for feature, importance in importances:
        if importance > 0:
            print(f"{feature:25s}{importance:.4f}")


# Pretty Tree Printer

def print_tree(tree, feature_names, node=0, indent=""):

    left = tree.tree_.children_left[node]
    right = tree.tree_.children_right[node]

    gini = tree.tree_.impurity[node]
    samples = tree.tree_.n_node_samples[node]

    proportions = tree.tree_.value[node][0]

    counts = proportions * samples

    majority = tree.classes_[proportions.argmax()]

    if left == -1:

        print(indent + "LEAF")
        print(indent + f"Prediction : {majority}")
        print(indent + f"Gini       : {gini:.3f}")
        print(indent + f"Samples    : {samples}")

        print(indent + "Class Counts:")

        for cls, cnt in zip(tree.classes_, counts):
            print(indent + f"   {cls:10s}: {int(round(cnt))}")

        return

    feature = feature_names[tree.tree_.feature[node]]
    threshold = tree.tree_.threshold[node]

    print()
    print(indent + f"Split: {feature} <= {threshold:.2f}")
    print(indent + f"Gini       : {gini:.3f}")
    print(indent + f"Samples    : {samples}")
    print(indent + f"Majority   : {majority}")

    print(indent + "Class Counts:")

    for cls, cnt in zip(tree.classes_, counts):
        print(indent + f"   {cls:10s}: {int(round(cnt))}")

    print(indent + f"├── YES (<= {threshold:.2f})")

    print_tree(tree, feature_names, left, indent + "│   ")

    print(indent + f"└── NO  (> {threshold:.2f})")

    print_tree(tree, feature_names, right, indent + "    ")


def main():

    X, y, feature_names = load_data()

    tree = train_tree(X, y)

    print()
    print()
    print("DECISION TREE")
    print()

    print_tree(tree, feature_names)

    print_feature_importance(tree, feature_names)


main()
