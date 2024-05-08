import pandas as pd
from pathlib import Path
from model_preprocessing import data_pre
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn
from joblib import dump, load # в scikit-learn ничего такого особенного нет - пользуемся joblib


filepath_train = r'dataset/train/df_train_1.csv'
df_train = data_pre(filepath_train)

# не забываем выделить целевую переменную из признаков X
X_train, Y_train = df_train.drop(columns = ['y']), df_train['y']

# Создаем модель
model_y = RandomForestRegressor(n_estimators=150, max_depth=10, oob_score=True)
model_y.fit(X_train, Y_train)

#Сохраняем модель
filepath_model = Path('model/model_y.joblib')
filepath_model.parent.mkdir(parents=True, exist_ok=True)
dump(model_y, filepath_model)  # чтобы сохранить объект

print('Модель обучена на тренировочных данных и сохранена')