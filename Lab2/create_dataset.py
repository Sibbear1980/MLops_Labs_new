from catboost.datasets import titanic
from pathlib import Path
import pandas as pd
import numpy as np


# загрузка данных
train, test = titanic()
#print(train.info())
pd.set_option('display.max_columns', None)
#print(train.head())
#print(train.drop_duplicates(['Embarked'])[['Embarked']])
#print(train.groupby('Embarked').agg({'PassengerId':'count'}))

# обработка данных
#features_all = list(train.columns[2:])
#print(features_all)
features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
features_all_wo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# заполним данные о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
train['Sex'] = train['Sex'].apply(lambda x: 0 if 'male' else 1)
test['Sex'] = test['Sex'].apply(lambda x: 0 if 'male' else 1)
# в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
train['Age'] = train['Age'].fillna(train.Age.mean())
test['Age'] = test['Age'].fillna(train.Age.mean())
# в признаке "Посадка" числовыми данными (0, 1, 2, 3) вместо текстовых
train['Embarked'] = train['Embarked'].apply(lambda x: 1 if 'C' else ('2' if 'S' else ('3' if 'Q' else 0)))
test['Embarked'] = test['Embarked'].apply(lambda x: 1 if 'C' else ('2' if 'S' else ('3' if 'Q' else 0)))

# запишем созданные датасеты во внешние csv-файлы
filepath_train = Path('data/data_train.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
train[features_all].to_csv(filepath_train, index=False)
filepath_test = Path('data/data_test.csv')
filepath_test.parent.mkdir(parents=True, exist_ok=True)
test[features_all_wo].to_csv(filepath_test, index=False)