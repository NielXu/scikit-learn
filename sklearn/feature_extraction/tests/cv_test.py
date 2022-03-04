from mmap import ALLOCATIONGRANULARITY
from unittest import expectedFailure
import pytest 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


def test_1():
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

def test_2():
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

def test_3():
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

def test_4():
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
