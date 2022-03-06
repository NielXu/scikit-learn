from mmap import ALLOCATIONGRANULARITY
from unittest import expectedFailure
import pytest 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


def test_laacFalse_withTrademark():
    x = ['This is Problematic™.','THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,1),
        lowercase_after_accent_functions=False
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematicTM', 'this']
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_laacTrue_withTrademark():
    x = ['This is Problematic™.','THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,1),
        lowercase_after_accent_functions=True
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematictm', 'this']
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_laacTrue_withoutTrademark():
    x = ['This is ProblematicTM.','THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,1),
        lowercase_after_accent_functions=True
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematictm', 'this']
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

def test_laacFalse_withoutTrademark():
    x = ['This is ProblematicTM.','THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,1),
        lowercase_after_accent_functions=False
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematictm', 'this']
    
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    




if __name__ == '__main__':
    pytest.main()
