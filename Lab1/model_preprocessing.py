import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_pre (path):
    #Загружаем датафрейм
    df = pd.read_csv(path, encoding='utf-8', sep=',')
    col_names = ['x1', 'x2', 'x3', 'x4', 'x5']

    #Выделяем целевую функцию
    Y = df.copy('y')
    X = df.drop('y', axis=1)

    #Выполняем стандартизацию данных
    f_stand = StandardScaler()
    f_stand.fit(X)

    #Применяем трансформер
    f_standarted = f_stand.transform(X[col_names])
    X_stand = pd.DataFrame(f_standarted, columns=col_names)

    #Масштабируем данные
    f_scale = MinMaxScaler()
    f_scale.fit(X_stand)

    #Применяем трансформер
    f_scaled = f_scale.transform(X_stand[col_names])
    X_prepared = pd.DataFrame(f_scaled, columns=col_names)

    X_prepared = pd.concat([X_prepared, Y], axis=1)

    return X_prepared

print('Функция подготовки данных загружена')