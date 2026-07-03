import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)

# Set the parameters for this run
d = 2
r = 1
C = 1

# Get the data
INPUT_FILE = "Demo_Data_Little.csv"
df = pd.read_csv(INPUT_FILE)
df["Played in 2001"] = df["Played in 2001"].map({"Yes": 1, "No": 0})
X = df[["Rushing Yards 2000","Fumbles 2000"]].values
y = df["Played in 2001"].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Support Vector Machine
svm = SVC(kernel="poly", degree=d, coef0=r, C=C, gamma="scale")
svm.fit(X_scaled, y)
pred = svm.predict(X_scaled)

# Create the grid
xmin = df["Rushing Yards 2000"].min() - 50
xmax = df["Rushing Yards 2000"].max() + 50
ymin = df["Fumbles 2000"].min() - 1
ymax = df["Fumbles 2000"].max() + 1
xx = np.linspace(xmin, xmax, 500)
yy = np.linspace(ymin, ymax, 500)
YY, XX = np.meshgrid(yy, xx)
grid = np.c_[XX.ravel(), YY.ravel()]
grid_scaled = scaler.transform(grid)
Z = svm.decision_function(grid_scaled)
Z = Z.reshape(XX.shape)

# Gather what we need to plot it
plt.figure(figsize=(11,8))

# Ready the background shading
plt.contourf(XX, YY, Z, levels=[-1000,0,1000], alpha=.18,
             colors=["firebrick","forestgreen"])

# Plot the decision boundary
plt.contour(XX, YY, Z, levels=[-1,0,1], colors=["navy","blue","navy"],
            linestyles=["--","-","--"], linewidths=2)

# Plot the data points

played = df[df["Played in 2001"]==1]
retired = df[df["Played in 2001"]==0]
plt.scatter(played["Rushing Yards 2000"], played["Fumbles 2000"], s=120,
            color="forestgreen", edgecolors="black", label="Played in 2001")
plt.scatter(retired["Rushing Yards 2000"], retired["Fumbles 2000"], s=120,
            color="firebrick", edgecolors="black", label="Retired")

# Label each point
for i,row in df.iterrows():
    plt.text(row["Rushing Yards 2000"]+15, row["Fumbles 2000"]+.08,
             row["Label Num"], fontsize=9)

# Draw the support vectors
support_vectors = scaler.inverse_transform(svm.support_vectors_)

plt.scatter(support_vectors[:,0], support_vectors[:,1], s=300,
            facecolors="none", edgecolors="gold",
            linewidths=2.5, label="Support Vectors")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel("Rushing Yards (2000)")
plt.ylabel("Fumbles (2000)")
plt.title(
    f"Polynomial SVM\n"
    f"d={d}, r={r}, "
    f"Accuracy={accuracy_score(y,pred):.3f}"
)

plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()

# Finally plot the whole thing
plt.show()
