import csv

INPUT_2000 = "NFL_Rushing_2000.csv"
INPUT_2001 = "NFL_Rushing_2001.csv"
OUTPUT_FILE = "NFL_2D_Demo_Data.csv"

players_2001 = set()

with open(INPUT_2001, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        players_2001.add(row["Name"].strip())

output_rows = []

with open(INPUT_2000, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        name = row["Name"].strip()
        rushing_yards = row["Rushing Yards"].strip()
        fumbles = row["Fumbles"].strip()
        played_2001 = "Yes" if name in players_2001 else "No"
        output_rows.append([name, rushing_yards, fumbles, played_2001])

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Name", "Rushing Yards 2000",
                     "Fumbles 2000", "Played in 2001"])
    writer.writerows(output_rows)


print(f"Wrote {len(output_rows)} players to {OUTPUT_FILE}")
