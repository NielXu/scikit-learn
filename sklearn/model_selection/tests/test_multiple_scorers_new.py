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

def test_without_scoring_handle_errors_False():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00049853, 0.00031328, 0.00029945, 0.0002985 , 0.00029778]), 
                'score_time': np.array([0.00023556, 0.00020146, 0.00020027, 0.00019979, 0.00019908]),
                'test_score': np.array([0.98185992, 0.98455702, 0.98816009, 0.98852095, 0.99761839])}

    actual = cross_validate(linear_model.LinearRegression(),X,y, handle_errors = False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_without_scoring_handle_errors_True():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00049853, 0.00031328, 0.00029945, 0.0002985 , 0.00029778]), 
                'score_time': np.array([0.00023556, 0.00020146, 0.00020027, 0.00019979, 0.00019908]),
                'test_score': np.array([0.98185992, 0.98455702, 0.98816009, 0.98852095, 0.99761839])}

    actual = cross_validate(linear_model.LinearRegression(),X,y, handle_errors = True)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_r2():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    ) 
    expected = {'fit_time': np.array([0.00025868, 0.00024986, 0.00023913, 0.00023603, 0.00023198]), 
                'score_time': np.array([0.00016618, 0.00015736, 0.00015283, 0.00015569, 0.00015402]), 
                'test_score': np.array([0.98185992, 0.98455702, 0.98816009, 0.98852095, 0.99761839])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "r2", handle_errors=False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_neg_median_absolute_error():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00024128, 0.00023103, 0.00023985, 0.00022101, 0.00022388]), 
                'score_time': np.array([0.00016475, 0.00014615, 0.00013781, 0.00014496, 0.00013018]), 
                'test_score': np.array([ -7.40444967,  -6.9596981 ,  -5.83755438, -10.1810846 ,
        -4.96515603])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "neg_median_absolute_error", handle_errors=False)
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_neg_mean_absolute_error():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00024509, 0.00022721, 0.0002203 , 0.00022507, 0.00021791]), 
                'score_time': np.array([0.00012589, 0.00012279, 0.00012279, 0.00012112, 0.00011587]), 
                'test_score': np.array([-8.88891487, -7.57388138, -7.15079858, -9.84652849, -5.06283734])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "neg_mean_absolute_error", handle_errors=False)
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_neg_mean_absolute_percentage_error():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.0002439 , 0.00023007, 0.00022292, 0.00022507, 0.00021768]), 
                'score_time': np.array([0.0001421 , 0.00012589, 0.00012112, 0.00012207, 0.00012231]), 
                'test_score': np.array([-0.20011721, -0.18335903, -0.17779721, -1.02021875, -0.11819356])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "neg_mean_absolute_percentage_error", handle_errors=False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_neg_mean_squared_log_error():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00023913, 0.00025129, 0.00023198, 0.00022388, 0.00022197]), 
                'score_time': np.array([0.00073862, 0.00017881, 0.00016618, 0.00016308, 0.0001657 ]), 
                'test_score': np.array([np.nan, np.nan, np.nan, np.nan, np.nan])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "neg_mean_squared_log_error", handle_errors=False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_scoring_string_neg_root_mean_squared_error():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=2,
        n_informative=1,
        noise=10,
        random_state=0,
    )
    expected = {'fit_time': np.array([0.00024796, 0.00023293, 0.00024796, 0.00022197, 0.0002172 ]), 
                'score_time': np.array([0.00012708, 0.00014997, 0.00011706, 0.00011802, 0.00011706]), 
                'test_score': np.array([-11.63580534, -10.24351287,  -8.82367169, -11.5066619 ,
        -6.03460018])}

    actual = cross_validate(linear_model.LinearRegression(), X,y,scoring = "neg_root_mean_squared_error", handle_errors=False)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

if __name__ == '__main__':
    pytest.main()