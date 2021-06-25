import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_recall_fscore_support
from titanic_service.model.preprocessor import Preprocessor
from typing import Dict, Union


class TitanicParameters(BaseModel):
    name: str
    AdaBoostClassifier: dict
    GradientBoostingClassifier: dict
    RandomForestClassifier: dict


class TitanicPredict(BaseModel):
    model_id: int
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str


class TitanicModel:
    TARGET = 'Survived'
    TRAIN_PATH = 'model/input/train_split.csv'
    TEST_PATH = 'model/input/test_split.csv'
    PREDICTIONS_CLASSES = {
        0: 'Survived',
        1: 'Did not survive'
    }

    def __init__(self):
        self.model = None
        self.metrics = None
        self._train_dataset = None
        self._test_dataset = None

    @property
    def train_dataset(self) -> pd.DataFrame:
        if self._train_dataset is None:
            self._train_dataset = pd.read_csv(self.TRAIN_PATH)
        return self._train_dataset

    @property
    def test_dataset(self) -> pd.DataFrame:
        if self._test_dataset is None:
            self._test_dataset = pd.read_csv(self.TEST_PATH)
        return self._test_dataset

    def train(self, parameters: dict) -> "TitanicModel":
        model = self.build(parameters)
        self.model = model.fit(
            self.train_dataset.drop(columns=[self.TARGET]),
            self.train_dataset[self.TARGET]
        )
        self.score_model()
        return self

    def predict(self, data: pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        return self.model.predict(data)

    @staticmethod
    def build_classifier(parameters: dict) -> VotingClassifier:
        return VotingClassifier(estimators=[
            ('AdaBoostClassifier', AdaBoostClassifier(**parameters.get('AdaBoostClassifier', {}))),
            ('GradientBoostingClassifier', GradientBoostingClassifier(**parameters.get('GradientBoostingClassifier', {}))),
            ('RandomForestClassifier', RandomForestClassifier(**parameters.get('RandomForestClassifier', {})))
        ])

    def build(self, parameters: dict) -> Pipeline:
        transformer_fn = Preprocessor.build_transformer()
        transformer = FunctionTransformer(transformer_fn, validate=False)
        return Pipeline([
            ('preprocessing', transformer),
            ('voting_cls', self.build_classifier(parameters))
        ])

    def scoring(self, actual: Union[np.ndarray, pd.Series], predictions: Union[np.ndarray, pd.Series]) -> Dict:
        precisions, recalls, f_scores, _ = precision_recall_fscore_support(actual, predictions)
        return {
            'precision_survived': precisions[0],
            'precision_didnt_survive': precisions[1],
            'recall_survived': recalls[0],
            'recall_didnt_survive': recalls[1],
            'f_score_survived': f_scores[0],
            'f_score_didnt_survive': f_scores[1]
        }

    def score_model(self):
        self.metrics = self.scoring(
            actual=self.test_dataset[self.TARGET],
            predictions=self.model.predict(self.test_dataset.drop(columns=[self.TARGET]))
        )
