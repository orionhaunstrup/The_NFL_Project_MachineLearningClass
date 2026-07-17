"""

THis program splits the NN training data into testing and training

"""

import csv
import os
import random

for filename in sorted(os.listdir(".")):

    if not filename.endswith(".csv"):
        continue

    print(f"Processing {filename}...")

    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    if len(reader) <= 1:
        print("  Skipped (empty dataset)")
        continue

    header = reader[0]
    rows = reader[1:]

    # Randomize player order
    random.shuffle(rows)

    # 80 / 20 split
    split_index = int(0.80 * len(rows))

    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    train_filename = "Train" + filename
    test_filename = "Test" + filename

    with open(train_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    with open(test_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    # Delete original dataset
    os.remove(filename)
