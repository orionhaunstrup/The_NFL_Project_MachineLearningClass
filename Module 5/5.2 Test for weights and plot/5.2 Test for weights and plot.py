"""

This program is both the first NN attempt and also a means
of testing for what is the best retirement weight value


"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Ready the NN values

TRAIN_FILE = "TrainQB5.csv"
TEST_FILE  = "TestQB5.csv"

EPOCHS = 50
LEARNING_RATE = 0.001

# Read in all the data from the .csv files

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.iloc[:,1:-2].values
X_test  = test_df.iloc[:,1:-2].values

y_train = train_df["Retire"].values
y_test  = test_df["Retire"].values

# Storage vectors, for plotting later

weights = []

accuracies = []

true_positives = []
false_positives = []

true_negatives = []
false_negatives = []


print()
print()
print("Retirement Weight Sweep")
print()

print()
print(f"{'Weight':<8}{'Accuracy':<12}{'TP':<6}{'TN':<6}{'FP':<6}{'FN':<6}")
print()

for RETIRE_WEIGHT in range(1,11):
    
    # Ensuring that all else BUT the retirement weights stays the same
    np.random.seed(333)
    tf.random.set_seed(333)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    # The first neurons
    model.add(Dense(units=16, activation="relu"))
    # The final neuron
    model.add(Dense(units=1, activation="sigmoid"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                  metrics=["accuracy"])
    # Adding the retirement weight into the picture
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0,
              class_weight={0:1, 1:RETIRE_WEIGHT})
    probabilities = model.predict(X_test, verbose=0)
    predictions = (probabilities > 0.5).astype(int)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    weights.append(RETIRE_WEIGHT)

    accuracies.append(acc)

    true_positives.append(tp)
    false_positives.append(fp)

    true_negatives.append(tn)
    false_negatives.append(fn)

    print(f"{RETIRE_WEIGHT:<8}{acc:0.4f}      {tp:<6}{tn:<6}{fp:<6}{fn:<6}")
    print()
    print(cm)
    print()

## Plot the accuracies

plt.figure(figsize=(8,5))
plt.plot(weights, accuracies, marker='o', linewidth=2,
         color = "xkcd:neon pink")
plt.xlabel("Retirement Weight")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Retirement Weight")
plt.xticks(weights)
plt.grid(True)
plt.tight_layout()
plt.savefig("AccuracyVsWeight.png")
plt.close()

## Plot the confusion matrix statistics

plt.figure(figsize=(8,5))
plt.plot(weights, true_positives, marker='o', linewidth=2,
         label="Correct Retirements (TP)", color = "xkcd:ocean")
plt.plot(weights, false_positives, marker='s', linewidth=2,
         label="False Retirements (FP)", color = "xkcd:orange")
plt.plot(weights, false_negatives, marker='^', linewidth=2,
         label="Missed Retirements (FN)",color = "xkcd:purple")
plt.xlabel("Retirement Weight")
plt.ylabel("Players")
plt.title("Effect of Retirement Weight on Retirement Predictions")
plt.xticks(weights)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ConfusionTradeoff.png")
plt.close()
