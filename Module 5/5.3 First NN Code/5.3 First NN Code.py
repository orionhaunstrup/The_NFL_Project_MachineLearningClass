"""

Early NN version, to test for accuracy and confusion matrices.
Only tests for retirement

"""

import os

import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import random
import matplotlib.pyplot as plt


POSITIONS = [

    "QB",
    "Rushing",
    "Receiving",
    "Defense",
    "Kicking"

]

YEARS = range(3,11)

EPOCHS = 50

LEARNING_RATE = 0.001

RETIRE_WEIGHT = 4

models = {}

test_sets = {}

histories = {}

## Now to train all the datasets

for position in POSITIONS:

    for years in YEARS:

        TRAIN_FILE = f"Train{position}{years}.csv"
        TEST_FILE  = f"Test{position}{years}.csv"

        if not os.path.exists(TRAIN_FILE):
            continue

        if not os.path.exists(TEST_FILE):
            continue

        print()
        print()
        print(f"Training {position} {years}")
        print()

        # Read in the data
        train_df = pd.read_csv(TRAIN_FILE)
        test_df  = pd.read_csv(TEST_FILE)

        # Features
        X_train = train_df.iloc[:,1:-2].values

        # Do they or do they not retire
        y_train = train_df["Retire"].values

        ## Now to build all the NNs
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=["accuracy"])

        # Train
        history = model.fit(X_train, y_train, pochs=EPOCHS, verbose=0,
                            class_weight={0:1, 1:RETIRE_WEIGHT})

        # Save everything
        key = (position, years)
        models[key] = model
        test_sets[key] = test_df
        histories[key] = history

# Print a summary

print()
print()
print("Training Complete")
print()
print("Datasets Trained:", len(models))
print()

for key in sorted(models.keys()):
    print(key)








# Build the master test list
master_test_list = []
for key in test_sets:
    test_df = test_sets[key]
    for row_index in range(len(test_df)):
        master_test_list.append((key, row_index))



# This is a little bit spaghetti-like, but should do the job


NUM_EXPERIMENTS = 10000
master_cm = [[0,0], [0,0]]
correct_predictions = 0
last_history = None



for experiment in range(NUM_EXPERIMENTS):

    # progress bar
    if (experiment+1) % 1000 == 0:

        print(experiment+1, "predictions complete.")

    key, row_index = random.choice(master_test_list)
    model = models[key]
    test_df = test_sets[key]
    row = test_df.iloc[[row_index]]
    X = row.iloc[:,1:-2].values
    actual = int(row["Retire"].iloc[0])
    probability = model.predict(X, verbose=0)[0][0]
    predicted = int(probability > 0.5)
    if predicted == actual:
        correct_predictions += 1
    master_cm[actual][predicted] += 1


# Save the full experiment
last_history = histories[key]



accuracy = correct_predictions / NUM_EXPERIMENTS


print()
print()
print("FINAL RESULTS")
print()
print()
print("Experiments:", NUM_EXPERIMENTS)
print()
print("Accuracy:", round(accuracy,4))
print()
print("Master Confusion Matrix")
print(master_cm)




print()
print()
print("Last Five Epochs")
print()

for i in range(EPOCHS-5,EPOCHS):
    print("Epoch", i+1, "Loss =", round(last_history.history["loss"][i],6),
          "Accuracy =", round(last_history.history["accuracy"][i],6))


from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=master_cm)
disp.plot(cmap="Blues", values_format="d")

plt.title(f"Confusion Matrix\nAccuracy = {accuracy:.2%}")
plt.tight_layout()
plt.savefig("ConfusionMatrix.png")
