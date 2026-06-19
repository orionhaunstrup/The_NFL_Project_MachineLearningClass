import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

FILENAMES = [
    "DefenseDataCube_PCAReady.csv",
    "KickingDataCube_PCAReady.csv",
    "QBDataCube_PCAReady.csv",
    "ReceivingDataCube_PCAReady.csv",
    "RushingDataCube_PCAReady.csv"
]

for filename in FILENAMES:

    print()
    print(f"Processing {filename}")

    df = pd.read_csv(filename)

    #
    # Save player names
    #

    player_names = df.iloc[:, 0]

    #
    # Numerical matrix
    #

    X = df.iloc[:, 1:]

    #
    # Standardize
    #

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    #
    # SVD
    #

    U, S, VT = np.linalg.svd(
        X_scaled,
        full_matrices=False
    )

    Sigma = np.diag(S)

    base_name = filename.replace(
        "_Transformed2_PCAReady.csv",
        ""
    )

    #
    # U
    #

    U_df = pd.DataFrame(U)

    U_df.insert(
        0,
        "PLAYER NAME",
        player_names
    )

    U_df.to_csv(
        f"{base_name}_U.csv",
        index=False
    )

    #
    # Sigma
    #

    pd.DataFrame(
        Sigma
    ).to_csv(
        f"{base_name}_Sigma.csv",
        index=False
    )

    #
    # VT
    #

    pd.DataFrame(
        VT
    ).to_csv(
        f"{base_name}_VT.csv",
        index=False
    )

    print(
        f"Players: {U.shape[0]}"
    )

    print(
        f"Components: {U.shape[1]}"
    )

    print(
        f"Saved {base_name}_U.csv"
    )

    print(
        f"Saved {base_name}_Sigma.csv"
    )

    print(
        f"Saved {base_name}_VT.csv"
    )

print()
print("Done.")
