import pickle
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import ShuffleSplit # при кросс-валидации случайно перемешиваем данные
from sklearn.model_selection import cross_validate # функция кросс-валидации от Scikit-learn

features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
features_all_wo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#Зададим функцию кросс-валидации
def cross_validation (X, y, model, scoring, cv_rule):
    """Расчет метрик на кросс-валидации.
    Параметры:
    ===========
    model: модель или pipeline
    X: признаки
    y: истинные значения
    scoring: словарь метрик
    cv_rule: правило кросс-валидации
    """
    scores = cross_validate(model,X, y,
                      scoring=scoring, cv=cv_rule )
    print('Ошибка на кросс-валидации')
    DF_score = pd.DataFrame(scores)
    display(DF_score)
    print('\n')
    print(DF_score.mean()[2:])

filepath_model = r'model/model_tk.pkl'
loaded_model = pickle.load(open(filepath_model, 'rb'))

# прочитаем из csv-файла подготовленный датасет для обучения
filepath_test = r'data/data_test.csv'
data_test = pd.read_csv(filepath_test)
X_test = data_test[features_all_wo].values
Y_predict = loaded_model.predict(X_test)
# сделаем предсказание для первого пассажира из тестовой выборки
print('Предсказание для первого пассажира', loaded_model.predict(X_test[0:1]))

#Сделаем кросс-валидацию и посчитаем метрики
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X_train = data_train[features_all_wo].values
Y_train = data_train['Survived'].values
scoring_reg = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           '-Max': 'max_error'}

cross_validation (X_train, Y_train,
                  loaded_model,
                  scoring_reg,
                  ShuffleSplit(n_splits=5, random_state = 42))

