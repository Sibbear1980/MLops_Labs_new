#Создайте python-скрипт (data_creation.py), который создает различные наборы данных,
# описывающие некий процесс (например, изменение дневной температуры).
# Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы.
# Часть наборов данных должна быть сохранена в папке «train», другая часть — в папке «test».
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons, make_blobs # генерируемые наборы данных
from sklearn.model_selection import train_test_split #  функция разбиения на тренировочную и тестовую выборку
import matplotlib.pyplot as plt # библиотека Matplotlib для визуализации
from pathlib import Path

# Генерим наборы данных - для примера 5 наборов
# общие параметры для данных в виде скоплений точек
n_samples = [100,1000,300,800,200] #Количество точек всего
outliers_fraction = 0.05 #доля шума и/или аномалий
j = 0 #Номер набора от которого будут зависить параметры набора (размер, коэффициенты регрессии, нулевая точка
for n in n_samples:
    print(n,j)
    n_outliers = int(outliers_fraction * n_samples[j]) # количество шума в штуках
    n_inliers = n_samples[j] - n_outliers #количество нормальных точек в штуках
    blobs_params = dict(random_state=42+j, n_samples=n_inliers, n_features=5)
    # Зададим распределение точек с центром в [0, 0] и СКО 0.5
    data_norm, _ = make_blobs(centers=[[0, 0, 0, 0, 0], ], cluster_std=0.5, **blobs_params)
    df = pd.DataFrame(data_norm)

    #Переименуем колонки
    df = df.set_axis(['x1', 'x2', 'x3', 'x4', 'x5'], axis=1)

    # Добавим функцию этих параметров
    df['y'] = -j * df['x1'] + j * df['x2'] + df['x3']**2 - (df['x4'] * df['x5'])
    #print(df)

    #Случайный шум
    rng = np.random.RandomState(42+j)
    noise = rng.uniform(low=-6, high=6, size=(n_outliers, 6))
    df_noise = pd.DataFrame(noise)
    df_noise = df_noise.set_axis(['x1', 'x2', 'x3', 'x4', 'x5', 'y'], axis=1)
    df = pd.concat([df, df_noise])
    #print(df)

    # разбиваем на тренировочную и валидационную
    df_train = pd.DataFrame(df).sample(frac=0.7, random_state=42+j)
    df_val = pd.DataFrame(df).drop(df_train.index)

    #Смотрим, что получилось
    #print(df_train.shape)
    #print(df_val.shape)
    #print(df_train)

    #Сохраняем датафреймы
    filepath_train = Path('dataset/train/df_train_'+str(j)+'.csv')
    filepath_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(filepath_train, index=False)

    filepath_test = Path('dataset/test/df_test_'+str(j)+'.csv')
    filepath_test.parent.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(filepath_test, index=False)

    j += 1

print('Набор датасетов создан и сохранен')


