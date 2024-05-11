import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn

features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
features_all_wo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# прочитаем из csv-файла подготовленный датасет для обучения
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X_train = data_train[features_all_wo].values
Y_train = data_train['Survived'].values

# загрузим модель машинного обучения. Будем использовать Случайный лес

model_tk = RandomForestRegressor(n_estimators=150, max_depth=10, oob_score=True)
model_tk.fit(X_train, Y_train)

# сохраним обученную модель
filepath_model = Path('model/model_tk.pkl')
filepath_model.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(model_tk, open(filepath_model, 'wb'))

print('Модель сохранена')