"""

This program builds the NN training sets

"""

from collections import defaultdict
import csv
import os

DATACUBES = {
    "QB": "QBDataCubeTrimmed.csv",
    "Rushing": "RushingDataCubeTrimmed.csv",
    "Receiving": "ReceivingDataCubeTrimmed.csv",
    "Defense": "DefenseDataCubeTrimmed.csv",
    "Kicking": "KickingDataCubeTrimmed.csv"
}


def load_players(filename):
    player_rows = defaultdict(list)
    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                player = row[0]
                percentile = float(row[1])
                player_rows[player].append(percentile)
    return player_rows


def build_training_sets(position_name, filename):
    players = load_players(filename)
    longest_career = max(len(career) for career in players.values())
    for years_revealed in range(3, longest_career + 1):
        output_filename = f"{position_name}{years_revealed}.csv"
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["Player"]
            for i in range(1, years_revealed + 1):
                header.append(f"Year{i}")
            header.append("Retire")
            header.append("NextPercentile")
            writer.writerow(header)
            rows_written = 0
            for player, career in sorted(players.items()):
                if len(career) < years_revealed:
                    continue
                row = [player]
                row.extend(career[:years_revealed])
                if len(career) == years_revealed:
                    row.append(1)
                    row.append("")
                else:
                    row.append(0)
                    row.append(career[years_revealed])
                writer.writerow(row)
                rows_written += 1
        print(f"  {output_filename:<15} {rows_written} rows")


def main():
    for position, filename in DATACUBES.items():
        build_training_sets(position, filename)
        print()

main()
