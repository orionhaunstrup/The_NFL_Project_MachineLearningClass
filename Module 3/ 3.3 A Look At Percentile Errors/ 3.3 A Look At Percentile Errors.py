import csv
import statistics


FILES = [
    "QB_DecisionTree_Data.csv",
    "Rushing_DecisionTree_Data.csv",
    "Receiving_DecisionTree_Data.csv",
    "Defense_DecisionTree_Data.csv",
    "Kicking_DecisionTree_Data.csv"
]


for filename in FILES:
    percentile_differences = []
    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            percentile_differences.append(
                float(row["MEAN_PERCENTILE_DIFFERENCE"]))

    percentile_differences.sort()

    print()
    print()
    print(filename)

    print(
        f"Total Cases: "
        f"{len(percentile_differences):,}"
    )

    print(
        f"Minimum: "
        f"{min(percentile_differences):.4f}"
    )

    print(
        f"25th Percentile: "
        f"{statistics.quantiles(percentile_differences, n=4)[0]:.4f}"
    )

    print(
        f"Median: "
        f"{statistics.median(percentile_differences):.4f}"
    )

    print(
        f"75th Percentile: "
        f"{statistics.quantiles(percentile_differences, n=4)[2]:.4f}"
    )

    print(
        f"Maximum: "
        f"{max(percentile_differences):.4f}"
    )

    print(
        f"Mean: "
        f"{statistics.mean(percentile_differences):.4f}"
    )

    print(
        f"Standard Deviation: "
        f"{statistics.stdev(percentile_differences):.4f}"
    )
