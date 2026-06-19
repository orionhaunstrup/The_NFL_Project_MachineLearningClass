import pandas as pd
import numpy as np

DATASETS = [
    "QB",
    "Rushing",
    "Receiving",
    "Defense",
    "Kicking"
]

VARIANCE_THRESHOLD = 0.95

for dataset in DATASETS:

    print()
    print(f"Processing {dataset}")

    #
    # Load matrices
    #

    U_df = pd.read_csv(
        f"{dataset}_U.csv"
    )

    Sigma_df = pd.read_csv(
        f"{dataset}_Sigma.csv"
    )

    VT_df = pd.read_csv(
        f"{dataset}_VT.csv"
    )

    #
    # Determine k
    #

    sigma_matrix = Sigma_df.to_numpy()

    singular_values = np.diag(
        sigma_matrix
    )

    explained_variance = (
        singular_values ** 2
        /
        np.sum(
            singular_values ** 2
        )
    )

    cumulative_variance = np.cumsum(
        explained_variance
    )

    k = (
        np.argmax(
            cumulative_variance
            >= VARIANCE_THRESHOLD
        )
        + 1
    )

    print(
        f"Keeping {k} components"
    )

    print(
        f"Cumulative variance = "
        f"{cumulative_variance[k-1]:.4f}"
    )

    #
    # Trim U
    #

    player_names = U_df.iloc[:, 0]

    U_numeric = (
        U_df.iloc[:, 1:]
    )

    U_trimmed = (
        U_numeric.iloc[:, :k]
    )

    U_trimmed.insert(
        0,
        "PLAYER NAME",
        player_names
    )

    #
    # Trim Sigma
    #

    Sigma_trimmed = (
        sigma_matrix[:k, :k]
    )

    #
    # Trim VT
    #

    VT_trimmed = (
        VT_df.iloc[:k, :]
    )

    #
    # Save
    #

    U_trimmed.to_csv(
        f"{dataset}_U_95Percent.csv",
        index=False
    )

    pd.DataFrame(
        Sigma_trimmed
    ).to_csv(
        f"{dataset}_Sigma_95Percent.csv",
        index=False
    )

    VT_trimmed.to_csv(
        f"{dataset}_VT_95Percent.csv",
        index=False
    )

    print(
        f"Saved {dataset}_U_95Percent.csv"
    )

    print(
        f"Saved {dataset}_Sigma_95Percent.csv"
    )

    print(
        f"Saved {dataset}_VT_95Percent.csv"
    )

print()
print("Done.")
