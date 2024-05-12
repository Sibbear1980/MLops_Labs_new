from catboost.datasets import titanic
from pathlib import Path
import pandas as pd

train, test = titanic()
features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']

# заполним данные о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
train['Sex'] = train['Sex'].apply(lambda x: 0 if 'male' else 1)
# в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
train['Age'] = train['Age'].fillna(train.Age.mean())
# в признаке "Посадка" числовыми данными (0, 1, 2, 3) вместо текстовых
train['Embarked'] = train['Embarked'].apply(lambda x: 1 if 'C' else ('2' if 'S' else ('3' if 'Q' else 0)))

# запишем созданные датасеты во внешние csv-файлы
filepath_train = Path('datasets/data_train.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
train[features_all].to_csv(filepath_train, index=False)