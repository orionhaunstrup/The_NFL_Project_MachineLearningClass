import csv
import statistics
from collections import Counter
import matplotlib.pyplot as plt


FILES = [
    "QB_DecisionTree_Data.csv",
    "Rushing_DecisionTree_Data.csv",
    "Receiving_DecisionTree_Data.csv",
    "Defense_DecisionTree_Data.csv",
    "Kicking_DecisionTree_Data.csv"
]


for filename in FILES:
    
    retirement_differences = []
    
    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            retirement_differences.append(int(row["RETIREMENT_DIFFERENCE"]))

    counts = Counter(retirement_differences)

    print()
    print(filename)

    total = len(retirement_differences)

    x_values = sorted(counts.keys())

    percentages = [
        100 * counts[x] / total
        for x in x_values
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(x_values, percentages, width=0.8)
    plt.title(filename.replace("_DecisionTree_Data.csv", ""))
    plt.xlabel("Retirement Difference")
    plt.ylabel("Percentage of Cases (%)")
    plt.xticks(x_values)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.show()
