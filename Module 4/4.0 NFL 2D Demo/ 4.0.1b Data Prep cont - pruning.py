import pandas as pd

# Interact with the files
INPUT_FILE = "NFL_2D_Demo_Data.csv"
OUTPUT_FILE = "Demo_Data_Little.csv"

# Read the dataset
df = pd.read_csv(INPUT_FILE)

# List of manually selected points
targets = [(132, 11), (1718, 5), (1361, 0), (1523, 4), (1369, 4),
           (500, 1), (294, 0), (16, 2), (22, 3), (161, 6), (475, 6)]

# Tolerance choices
yards_tol = 40
fumbles_tol = 0

selected_rows = []

# Keep only players within the selections
for _, row in df.iterrows():

    yards = row["Rushing Yards 2000"]
    fumbles = row["Fumbles 2000"]

    for target_yards, target_fumbles in targets:

        if (abs(yards - target_yards) <= yards_tol and
            abs(fumbles - target_fumbles) <= fumbles_tol):

            selected_rows.append(row)
            break
        
# Save the smaller dataset

selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv(OUTPUT_FILE, index=False)

print(selected_df)
print()
print(f"Selected {len(selected_df)} players.")
print(f"Saved to '{OUTPUT_FILE}'.")
