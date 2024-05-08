from pathlib import Path

import pandas as pd

from model_preprocessing import data_pre
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn
from joblib import dump, load # в scikit-learn ничего такого особенного нет - пользуемся joblib

#Выгружаем модель
filepath_model = Path('model/model_y.joblib')
model_y = load(filepath_model)

#Выгружаем тестовые данные
filepath_test = r'dataset/test/df_test_1.csv'
df_test = data_pre(filepath_test)

# не забываем выделить целевую переменную из признаков X
X_test, Y_test = df_test.drop(columns = ['y']), df_test['y']

# Применяем модель на тестовых данных
#model_y = RandomForestRegressor(n_estimators=150, max_depth=10, oob_score=True)
Y_target = model_y.predict(X_test)
Y_target = pd.DataFrame(Y_target, columns=['y_target'])
df_test_target = pd.concat([df_test, Y_target], axis=1)
#print(df_test_target)

print('Модель успешно апробирована на тестовых данных')
