import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Make the lists of dots we'll need
x = []
y = []
z = []

# Generate each dot
for layer in range(23):
    depth = random.randint(3, 23)
    for row in range(23):
        for col in range(depth):
            x.append(layer)
            y.append(row)
            z.append(col)

# Now let's plot it all
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')

ax.scatter(x, z, y, s=20)

ax.set_xlabel(" ")
ax.set_ylabel(" ")
ax.set_zlabel(" ")
ax.set_title("Jagged Cube")

plt.show()
