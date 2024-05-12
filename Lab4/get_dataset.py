from catboost.datasets import titanic
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)

train, test = titanic()
features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']

train1 = train.copy()
# заполним данные о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
train1['Sex'] = train1['Sex'].apply(lambda x: 1 if x == 'male' else (2 if x == 'female' else 0))
# в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
train1['Age'] = train1['Age'].fillna(train.Age.mean())
# в признаке "Посадка" числовыми данными (0, 1, 2, 3) вместо текстовых
train1['Embarked'] = train1['Embarked'].apply(lambda x: 1 if x == 'C' else ('2' if x == 'S' else ('3' if x == 'Q' else 0)))

# запишем созданные датасеты во внешние csv-файлы
filepath_train1 = Path('datasets/data_train1.csv')
#filepath_train1.parent.mkdir(parents=True, exist_ok=True)
train1[features_all].to_csv(filepath_train1, index=False)

#Сделать датасет с информацией только о классе, поле и возрасте
train2 = train.copy(['Pclass', 'Sex', 'Age'])
#print(train2)
#Save new dataset
filepath_train2 = Path('datasets/data_train2.csv')
#filepath_train2.parent.mkdir(parents=True, exist_ok=True)
train2.to_csv(filepath_train2, index=False)

#Сделать датасет с OneHotEncoder feature Sex
from sklearn.preprocessing import OneHotEncoder
train_OneHotEnc = train.copy()
# Transform the date
enc = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(enc.fit_transform(train[['Sex']]).toarray ())
train_OneHotEnc = train_OneHotEnc.join(encoder_df)
print(train_OneHotEnc.columns)
#Rename new created columns like logical names
train_OneHotEnc = train_OneHotEnc.rename(columns= {0 : 'Sex_is_male', 1 : 'Sex_isй_female'})
print(train_OneHotEnc.columns)

#Save new dataset
filepath_train3 = Path('datasets/data_train_OneHotEnc.csv')
#filepath_train3.parent.mkdir(parents=True, exist_ok=True)
train_OneHotEnc.to_csv(filepath_train3, index=False)

print('Данные сохранены')