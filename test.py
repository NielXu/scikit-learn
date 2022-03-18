import numpy as np
from sklearn.model_selection._split import TimeSeriesSplit, TimeSeriesSlidingWindow


X = np.arange(10)


print("TimeSeriesSplit is expanding window")
tscv = TimeSeriesSplit(n_splits=3, gap=2)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
print()


print("TimeSeriesSplit need to specify n_splits")
tscv = TimeSeriesSplit(n_splits=7, max_train_size=3, test_size=1)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
print()


print("TimeSeriesSlidingWindow does not require n_splits")
tssw = TimeSeriesSlidingWindow(train_size=3, test_size=1)
for train_index, test_index in tssw.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

print()
print("Test")
X = np.arange(10)
tscv = TimeSeriesSlidingWindow(gap=2)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
