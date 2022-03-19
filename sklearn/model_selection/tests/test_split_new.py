import numpy as np
from sklearn.model_selection._split import TimeSeriesSlidingWindow


def test_default_behavior():
    X = np.arange(10)
    tscv = TimeSeriesSlidingWindow()
    splits = tscv.split(X)

    train, test = next(splits)
    assert (train == np.array([0, 1])).all()
    assert (test == np.array([2, 3])).all()

    train, test = next(splits)
    assert (train == np.array([1, 2])).all()
    assert (test == np.array([3, 4])).all()

    train, test = next(splits)
    assert (train == np.array([2, 3])).all()
    assert (test == np.array([4, 5])).all()

    train, test = next(splits)
    assert (train == np.array([3, 4])).all()
    assert (test == np.array([5, 6])).all()

    train, test = next(splits)
    assert (train == np.array([4, 5])).all()
    assert (test == np.array([6, 7])).all()

    train, test = next(splits)
    assert (train == np.array([5, 6])).all()
    assert (test == np.array([7, 8])).all()

    train, test = next(splits)
    assert (train == np.array([6, 7])).all()
    assert (test == np.array([8, 9])).all()

    # make sure the iteration stopped
    try:
        train, test = next(splits)
    except StopIteration:
        assert True

def test_specify_train_size():
    X = np.arange(10)
    tscv = TimeSeriesSlidingWindow(train_size=4)
    splits = tscv.split(X)

    train, test = next(splits)
    assert (train == np.array([0, 1, 2, 3])).all()
    assert (test == np.array([4, 5])).all()

    train, test = next(splits)
    assert (train == np.array([1, 2, 3, 4])).all()
    assert (test == np.array([5, 6])).all()


    train, test = next(splits)
    assert (train == np.array([2, 3, 4, 5])).all()
    assert (test == np.array([6, 7])).all()

    train, test = next(splits)
    assert (train == np.array([3, 4, 5, 6])).all()
    assert (test == np.array([7, 8])).all()

    train, test = next(splits)
    assert (train == np.array([4, 5, 6, 7])).all()
    assert (test == np.array([8, 9])).all()

    # make sure the iteration stopped
    try:
        train, test = next(splits)
    except StopIteration:
        assert True

def test_specify_test_size():
    X = np.arange(10)
    tscv = TimeSeriesSlidingWindow(test_size=4)
    splits = tscv.split(X)

    train, test = next(splits)
    assert (train == np.array([0, 1])).all()
    assert (test == np.array([2, 3, 4, 5])).all()

    train, test = next(splits)
    assert (train == np.array([1, 2])).all()
    assert (test == np.array([3, 4, 5, 6])).all()


    train, test = next(splits)
    assert (train == np.array([2, 3])).all()
    assert (test == np.array([4, 5, 6, 7])).all()

    train, test = next(splits)
    assert (train == np.array([3, 4])).all()
    assert (test == np.array([5, 6, 7, 8])).all()

    train, test = next(splits)
    assert (train == np.array([4, 5])).all()
    assert (test == np.array([6, 7, 8, 9])).all()

    # make sure the iteration stopped
    try:
        train, test = next(splits)
    except StopIteration:
        assert True

def test_specify_gap():
    X = np.arange(10)
    tscv = TimeSeriesSlidingWindow(gap=2)
    splits = tscv.split(X)

    train, test = next(splits)
    assert (train == np.array([0, 1])).all()
    assert (test == np.array([4, 5])).all()

    train, test = next(splits)
    assert (train == np.array([1, 2])).all()
    assert (test == np.array([5, 6])).all()

    train, test = next(splits)
    assert (train == np.array([2, 3])).all()
    assert (test == np.array([6, 7])).all()

    train, test = next(splits)
    assert (train == np.array([3, 4])).all()
    assert (test == np.array([7, 8])).all()

    train, test = next(splits)
    assert (train == np.array([4, 5])).all()
    assert (test == np.array([8, 9])).all()

    # make sure the iteration stopped
    try:
        train, test = next(splits)
    except StopIteration:
        assert True


def test_different_combination():
    X = np.arange(10)
    tscv = TimeSeriesSlidingWindow(train_size=3, test_size=3, gap=2)
    splits = tscv.split(X)

    train, test = next(splits)
    assert (train == np.array([0, 1, 2])).all()
    assert (test == np.array([5, 6, 7])).all()

    train, test = next(splits)
    assert (train == np.array([1, 2, 3])).all()
    assert (test == np.array([6, 7, 8])).all()

    train, test = next(splits)
    assert (train == np.array([2, 3, 4])).all()
    assert (test == np.array([7, 8, 9])).all()

    # make sure the iteration stopped
    try:
        train, test = next(splits)
    except StopIteration:
        assert True
