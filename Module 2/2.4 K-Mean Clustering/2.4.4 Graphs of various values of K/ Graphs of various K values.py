import matplotlib.pyplot as plt

K_values = [5,10,15,20,25,30,35,40,45,50]

retirement_errors = [
    2.6270,
    3.2360,
    3.2160,
    3.7540,
    3.7600,
    3.7270,
    3.3940,
    3.3770,
    3.4770,
    3.2960
]

prediction_errors = [
    0.1615,
    0.1560,
    0.1330,
    0.1390,
    0.1370,
    0.1369,
    0.1303,
    0.1263,
    0.1265,
    0.1311
]

baseline_retirement = 1.8315
baseline_prediction = 0.1094


# -------------------------
# Retirement Error Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(
    K_values,
    retirement_errors,
    color='darkred',
    marker='o',
    linewidth=2,
    label='Clustering Retirement Error'
)

plt.axhline(
    baseline_retirement,
    color='black',
    linestyle='--',
    linewidth=2,
    label='Previous Machine (1.8315)'
)

plt.title('Retirement Error vs Number of Clusters')
plt.xlabel('K')
plt.ylabel('Mean Retirement Error')
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.legend()

plt.tight_layout()
plt.savefig(' RetirementErrorVsK.png')
plt.show()


# -------------------------
# Prediction Error Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(
    K_values,
    prediction_errors,
    color='turquoise',
    marker='o',
    linewidth=2,
    label='Clustering Prediction Error'
)

plt.axhline(
    baseline_prediction,
    color='black',
    linestyle='--',
    linewidth=2,
    label='Previous Machine (0.1094)'
)

plt.title('Prediction Error vs Number of Clusters')
plt.xlabel('K')
plt.ylabel('Mean Prediction Error')
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.legend()

plt.tight_layout()
plt.savefig(' PredictionErrorVsK.png')
plt.show()
