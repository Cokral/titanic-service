import pandas as pd
import pytest
from titanic_service.model.model import TitanicModel


@pytest.fixture
def actual() -> pd.Series:
    yield pd.Series([
        1, 0, 1, 1
    ])

@pytest.fixture
def predictions() -> pd.Series:
    yield pd.Series([
        1, 1, 0, 1
    ])

@pytest.fixture
def model():
    yield TitanicModel()


class TestModel:

    def test_scoring_returns_dict(self, actual, predictions, model):
        scoring = model.scoring(actual, predictions)
        assert isinstance(scoring, dict)

    def test_scoring_returns_correct_keys(self, actual, predictions, model):
        result = model.scoring(actual, predictions)
        expected_result = ['precision_survived', 'precision_didnt_survive', 'recall_survived', 'recall_didnt_survive',
                           'f_score_survived', 'f_score_didnt_survive']
        assert [actual == expected for actual, expected in zip(sorted(result.keys()), sorted(expected_result))]
