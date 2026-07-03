import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("Demo_Data_Little.csv")

# Split the data into two classes
played = df[df["Played in 2001"] == "Yes"]
retired = df[df["Played in 2001"] == "No"]

plt.figure(figsize=(10, 7))
plt.scatter(played["Rushing Yards 2000"], played["Fumbles 2000"],
            color="forestgreen", edgecolors="black",
            s=90, label="Played in 2001")
plt.scatter(retired["Rushing Yards 2000"], retired["Fumbles 2000"],
            color="firebrick", edgecolors="black",
            s=90, label="Retired After 2000")
plt.title("Running Backs (2000)", fontsize=16)
plt.xlabel("Rushing Yards (2000)", fontsize=13)
plt.ylabel("Fumbles (2000)", fontsize=13)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.xlim(-100, 1850)
plt.ylim(-1, 13)

plt.show()


