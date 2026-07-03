import matplotlib.pyplot as plt

# Polynomial Kernel Results

bins = [2, 3, 4, 5, 6, 7, 8, 9]

retirement_error = [1.78, 1.63, 1.65, 1.65, 1.31, 1.53, 1.54, 1.49]
percentile_error = [0.138, 0.147, 0.122, 0.100, 0.103, 0.109, 0.086, 0.100]

# Scale retirement error to fit on the same graph
retirement_error = [x / 10 for x in retirement_error]

plt.figure(figsize=(8,5))
plt.plot(bins, retirement_error, marker='o', linewidth=2,
         label="Retirement Error ÷ 10")
plt.plot(bins, percentile_error, marker='s', linewidth=2,
         label="Percentile Error")

plt.xlabel("Number of Percentile Bins")
plt.ylabel("Error")
plt.title("Polynomial Kernel: Effect of Number of Percentile Bins")

plt.xticks(bins)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
