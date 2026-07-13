import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Demo_Data.csv")

TARGET = "PREDICTION CATEGORY"
FEATURES = ["YEARS_OF_TEST_DATA", "PASSING YARDS", "COMPLETION %",
            "TOUCHDOWNS", "QB RATING"]

# Assign each feature its own color
feature_colors = {"YEARS_OF_TEST_DATA": "tab:blue",
                  "PASSING YARDS": "tab:orange",
                  "COMPLETION %": "tab:green",
                  "TOUCHDOWNS": "tab:red",
                  "QB RATING": "tab:purple"}


# This tiny function computes the gini impurity of any data subset
def gini(data):
    counts = data[TARGET].value_counts()
    probs = counts / len(data)
    return 1 - sum(probs ** 2)


# Compute the weighted gini impurity after each split
def split_gini(data, feature, threshold):

    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if len(left) == 0 or len(right) == 0:
        return None

    return (len(left)/len(data) * gini(left) + len(
        right)/len(data) * gini(right))


# Test every column and every pair of neighboring values
labels = []
gini_values = []
colors = []

for feature in FEATURES:
    values = sorted(df[feature].unique())
    for i in range(len(values)-1):
        threshold = (values[i] + values[i+1]) / 2
        g = split_gini(df, feature, threshold)
        if g is not None:
            labels.append(f"{feature}\n≤ {threshold:.1f}")
            gini_values.append(g)
            colors.append(feature_colors[feature])


# And plot it all!
plt.bar(range(len(gini_values)), gini_values, color=colors)

# Get rid of the x_axis labels (too cluttered)
plt.xticks([])
plt.ylabel("Weighted Gini Impurity")
plt.xlabel("Candidate Splits")
plt.title("Weighted Gini Impurity of Every Candidate Split")

# Make a legend
handles = [plt.Rectangle((0,0),1,1,color=c) for c in feature_colors.values()]
plt.legend(handles, feature_colors.keys(), title="Feature")

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
