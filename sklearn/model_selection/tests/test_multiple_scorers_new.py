from unittest import expectedFailure
import numpy as np
import warnings
import pytest

from sklearn import linear_model, datasets
from sklearn.model_selection import cross_validate

def test_default_behavior():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00042605, 0.00029707, 0.0002532 , 0.00024676, 0.00024605]),
                'score_time': np.array([0.00124383, 0.00060105, 0.0005672 , 0.00056481, 0.00055504]),
                'test_r2': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_median_absolute_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_absolute_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_absolute_percentage_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_squared_log_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_root_mean_squared_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = (
                "r2",
                "neg_median_absolute_error",
                "neg_mean_absolute_error",
                "neg_mean_absolute_percentage_error",
                "neg_mean_squared_log_error",
                "neg_root_mean_squared_error",
                ))

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_handle_errors_True():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )

    expected = {'fit_time': np.array([0.00042605, 0.00026584, 0.00026703, 0.00026703, 0.00026703]), 
                'score_time': np.array([0.00097299, 0.00058699, 0.0006001 , 0.00060797, 0.00073075]), 
                'test_r2': np.array([0.98185992, 0.98455702, 0.98816009, 0.98852095, 0.99761839]), 
                'test_neg_median_absolute_error': np.array([ -7.40444967,  -6.9596981 ,  -5.83755438, -10.1810846 , -4.96515603]), 
                'test_neg_mean_absolute_error': np.array([-8.88891487, -7.57388138, -7.15079858, -9.84652849, -5.06283734]), 
                'test_neg_mean_absolute_percentage_error': np.array([-0.20011721, -0.18335903, -0.17779721, -1.02021875, -0.11819356]), 
                'test_neg_root_mean_squared_error': np.array([-11.63580534, -10.24351287,  -8.82367169, -11.5066619 , -6.03460018]), 
                'test_neg_mean_squared_log_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = (
                "r2",
                "neg_median_absolute_error",
                "neg_mean_absolute_error",
                "neg_mean_absolute_percentage_error",
                "neg_mean_squared_log_error",
                "neg_root_mean_squared_error",
                ), handle_errors = True)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_handle_errors_False():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00042605, 0.00029707, 0.0002532 , 0.00024676, 0.00024605]),
                'score_time': np.array([0.00124383, 0.00060105, 0.0005672 , 0.00056481, 0.00055504]),
                'test_r2': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_median_absolute_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_absolute_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_absolute_percentage_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_mean_squared_log_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), 
                'test_neg_root_mean_squared_error': np.array([np.nan, np.nan, np.nan, np.nan, np.nan])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = (
                "r2",
                "neg_median_absolute_error",
                "neg_mean_absolute_error",
                "neg_mean_absolute_percentage_error",
                "neg_mean_squared_log_error",
                "neg_root_mean_squared_error",
                ), handle_errors=False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

if __name__ == '__main__':
    pytest.main()