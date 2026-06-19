import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

FILENAMES = [
    "QB_Sigma.csv",
    "Rushing_Sigma.csv",
    "Receiving_Sigma.csv",
    "Defense_Sigma.csv",
    "Kicking_Sigma.csv"
]

NUM_COMPONENTS_TO_SHOW = 10

colors = [
    "hotpink",
    "turquoise",
    "mediumorchid",
    "coral",
    "gold",
    "limegreen",
    "deepskyblue",
    "tomato",
    "violet",
    "darkorange"
]

for filename in FILENAMES:

    print(f"Processing {filename}")

    sigma_df = pd.read_csv(filename)

    sigma_matrix = sigma_df.to_numpy()

    singular_values = np.diag(sigma_matrix)

    explained_variance = (
        singular_values ** 2
        /
        np.sum(singular_values ** 2)
    )

    color = random.choice(colors)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(
            1,
            NUM_COMPONENTS_TO_SHOW + 1
        ),
        explained_variance[
            :NUM_COMPONENTS_TO_SHOW
        ],
        marker='o',
        markersize=10,
        linewidth=3,
        color=color
    )

    plt.xlabel(
        "Principal Component"
    )

    plt.ylabel(
        "Explained Variance Ratio"
    )

    plt.title(
        filename.replace(
            "_Sigma.csv",
            ""
        )
        + " Scree Plot"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_filename = (
        filename.replace(
            "_Sigma.csv",
            "_ScreePlot.png"
        )
    )

    plt.savefig(
        output_filename
    )

    plt.close()

    print(
        f"Saved {output_filename}"
    )

print()
print("Done.")
