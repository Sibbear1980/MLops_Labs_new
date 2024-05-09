import pandas as pd
import numpy as np
from pathlib import Path
import pickle #Библиотека для сохранения моделей
import matplotlib.pyplot as plt # библиотека Matplotlib для визуализации

#Зададим функцию визуализации значимости признаков
def feature_importance_plotter(model, features_names):
    """Отрисовка значимости признаков в виде горизонтальных столбчатых диаграмм.
    Параметры:
    model: модель
    features_names: список имен признаков
    """
    feature_importance = model.feature_importances_
    sorted = np.argsort(feature_importance)
    ypos = np.arange(len(features_names))
    fig= plt.figure(figsize=(8,4))
    plt.barh(ypos, feature_importance[sorted])
    #plt.xlim([0,1])
    plt.ylabel('Параметры')
    plt.xlabel('Значимость')
    plt.yticks(ypos, features_names[sorted])
    plt.show()

features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
features_all_wo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# прочитаем из csv-файла подготовленный датасет для обучения
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X_train = data_train[features_all_wo].values
Y_train = data_train['Survived'].values

# загрузим модель машинного обучения Регрессии
#from sklearn.linear_model import LogisticRegression
#model_tk = LogisticRegression(max_iter=100_000).fit(X_train, Y_train)

# загрузим модель машинного обучения. Будем использовать Случайный лес
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn
model_tk = RandomForestRegressor(n_estimators=150, max_depth=10, oob_score=True)
model_tk.fit(X_train, Y_train)

# загрузим модель машинного обучения. Будем использовать бустинг от CatBoost
# from catboost import CatBoostRegressor, Pool
# print(features_all_wo)
# train_data_reg = Pool(
#     data=X_train,
#     label=Y_train,
#     feature_names=features_all_wo,
# )
# eval_data_reg = Pool(
#     data=X_train,
#     label=Y_train,
#     feature_names=features_all_wo
# )
# model_tk = CatBoostRegressor(iterations = 500,
#                            early_stopping_rounds=10,
#                            verbose = 100,
#                            depth = 3,
#                               objective  = 'MAE',
#                            eval_metric= 'MAE',
#                               random_state = 42
#                            )
# model_tk.fit(X=train_data_reg,
#           eval_set=eval_data_reg,
#           )
feature_importance_plotter(model_tk, np.array(features_all_wo))
#С помощью графического отображения видно признаки, оказывающие влияние на целевую функцию

# сохраним обученную модель
filepath_model = Path('model/model_tk.pkl')
filepath_model.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(model_tk, open(filepath_model, 'wb'))