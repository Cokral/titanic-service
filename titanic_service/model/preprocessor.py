import pandas as pd
from typing import Callable


class Preprocessor:

    REQUIRED_COLUMNS = ['Pclass', 'Age', 'Fare', 'Master', 'Miss', 'Mr', 'Mrs', 'Rare', 'has_cabin', 'embarked_s',
                        'embarked_q', 'embarked_c', 'sex_male', 'family_size', 'is_alone']

    @classmethod
    def add_missing_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        new_columns = set(cls.REQUIRED_COLUMNS) - set(df.columns.unique())
        for new_col in new_columns:
            df[new_col] = 0
        return df

    @classmethod
    def remove_extra_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        extra_columns = set(df.columns.unique()) - set(cls.REQUIRED_COLUMNS)
        return df.drop(columns=extra_columns)

    @staticmethod
    def _prepare_status_passenger(series: pd.Series) -> pd.Series:
        series = series.apply(lambda x: x.split('.')[0].split(' ')[-1])
        series = series.replace(
            {'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss'}
        )
        series = series.replace(
            ['Dona', 'Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady', 'Capt', 'Countess', 'Jonkheer', 'Don'],
            'Rare'
        )
        return series

    @classmethod
    def prepare_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['status_passenger'] = cls._prepare_status_passenger(df.Name)
        df = df.join(pd.get_dummies(df['status_passenger']))

        df['has_cabin'] = df.Cabin.notna()
        df['Age'] = df.Age.fillna(df.Age.median())
        df['Fare'] = df.Fare.fillna(df.Fare.median())

        df['embarked_s'] = df.Embarked == 'S'
        df['embarked_q'] = df.Embarked == 'Q'
        df['embarked_c'] = df.Embarked == 'C'

        df['sex_male'] = df.Sex == 'male'

        df['family_size'] = df.SibSp + df.Parch + 1
        df['is_alone'] = df.family_size == 1

        df = df.replace({True: 1, False: 0})
        df = cls.add_missing_columns(df)
        df = cls.remove_extra_columns(df)
        return df

    @classmethod
    def build_transformer(cls) -> Callable:
        return cls.prepare_dataset
