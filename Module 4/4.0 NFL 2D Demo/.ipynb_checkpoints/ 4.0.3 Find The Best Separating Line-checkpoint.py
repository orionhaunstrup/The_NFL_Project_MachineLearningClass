import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the data
INPUT_FILE = "Demo_Data_Little.csv"

# Read the data
df = pd.read_csv(INPUT_FILE)

labels = df["Played in 2001"].map({"Yes": 1, "No": 0}).values

x = df["Rushing Yards 2000"].values
y = df["Fumbles 2000"].values

# Create the bounding box
xmin = x.min()
xmax = x.max()

ymin = y.min()
ymax = y.max()

# Set up for the search
best_accuracy = -1

best_x0 = None
best_y0 = None
best_theta = None
best_flip = False

NUM_TRIALS = 50000

# Run the search
for _ in tqdm(range(NUM_TRIALS)):

    # Select a random point
    x0 = random.uniform(xmin, xmax)
    y0 = random.uniform(ymin, ymax)

    # Select a random angle
    theta = random.uniform(0, 2 * math.pi)

    dx = math.cos(theta)
    dy = math.sin(theta)

    correct1 = 0
    correct2 = 0

    for xi, yi, label in zip(x, y, labels):
        side = dx * (yi - y0) - dy * (xi - x0)
        prediction = 1 if side > 0 else 0
        if prediction == label:
            correct1 += 1
        if (1 - prediction) == label:
            correct2 += 1

    accuracy = max(correct1, correct2) / len(df)

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_x0 = x0
        best_y0 = y0
        best_theta = theta

        best_flip = (correct2 > correct1)

# Gather the best line equation
dx = math.cos(best_theta)
dy = math.sin(best_theta)

print()

if abs(dx) < 1e-12:

    print("Best Line")
    print("---------")
    print(f"x = {best_x0:.3f}")

else:

    m = dy / dx
    b = best_y0 - m * best_x0

    print("Best Line")
    print("---------")
    print(f"y = {m:.6f}x + {b:.6f}")

# Get the final accuracy
correct = 0

played_correct = 0
played_total = 0

retired_correct = 0
retired_total = 0

for xi, yi, label in zip(x, y, labels):
    side = dx * (yi - best_y0) - dy * (xi - best_x0)
    prediction = 1 if side > 0 else 0
    if best_flip:
        prediction = 1 - prediction
    if prediction == label:
        correct += 1
    if label == 1:
        played_total += 1
        if prediction == 1:
            played_correct += 1
    else:
        retired_total += 1
        if prediction == 0:
            retired_correct += 1

# Print statements
print()
print("Results")
print("-------")
print(f"Correct: {correct} / {len(df)}")
print(f"Accuracy: {100*correct/len(df):.1f}%")
print()
print(f"Played correctly:  {played_correct}/{played_total}")
print(f"Retired correctly: {retired_correct}/{retired_total}")

# Ready the plot
plt.figure(figsize=(10,7))

# Shade the regions
xx = np.linspace(xmin-50, xmax+50, 500)
yy = np.linspace(ymin-1, ymax+1, 500)
XX, YY = np.meshgrid(xx, yy)
SIDE = dx * (YY - best_y0) - dy * (XX - best_x0)
prediction_grid = (SIDE > 0)
if best_flip:
    prediction_grid = ~prediction_grid
plt.contourf(XX, YY, prediction_grid.astype(int), levels=[-0.5,0.5,1.5],
             colors=["mistyrose","honeydew"], alpha=0.4)

# Plot it all
played = df[df["Played in 2001"] == "Yes"]
retired = df[df["Played in 2001"] == "No"]
plt.scatter(played["Rushing Yards 2000"], played["Fumbles 2000"],
            color="forestgreen", edgecolors="black", s=120,
            label="Actually Played")
plt.scatter(retired["Rushing Yards 2000"], retired["Fumbles 2000"],
            color="firebrick", edgecolors="black", s=120,
            label="Actually Retired")
L = 5000
x1 = best_x0 - dx * L
y1 = best_y0 - dy * L
x2 = best_x0 + dx * L
y2 = best_y0 + dy * L
plt.plot([x1,x2], [y1,y2], color="blue", linewidth=3,
         label="Best Random Separator")

# Make the labels
plt.text(xmin+60, ymax-0.6, "Predicted\nPlayed", fontsize=11,
         color="darkgreen", weight="bold")
plt.text(xmin+60, ymin+0.4, "Predicted\nRetired", fontsize=11,
         color="darkred", weight="bold")
plt.xlim(xmin-50, xmax+50)
plt.ylim(ymin-1, ymax+1)
plt.xlabel("Rushing Yards (2000)")
plt.ylabel("Fumbles (2000)")
plt.title(f"Best Random Linear Separator\nAccuracy = {100*best_accuracy:.1f}%")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()

# Generate the plot
plt.show()
