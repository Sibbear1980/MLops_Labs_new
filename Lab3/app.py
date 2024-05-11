import pickle
import pandas as pd
import streamlit as st

#streamlit run app.py --server.port=8501 server.address=0.0.0.0
#CMD ["streamlit", "run", "app.py","--server.port=8501", "server.address=0.0.0.0"]
#cd /Users/shabanovdmitry/Документы/УРФУ/MLops/Labs/Lab3
#docker image build -t live_titanic:0.1 .
#docker container run live_titanic:0.1
#Рабочий вариант запуска http://localhost:8501
#docker container run -p 8501:8501 live_titanic:0.1
#Тоже рабочий вариант запуска - открывать ссылку нужно из дэшборда докера http://localhost:8501
#docker container run -d -p 8501:8501 live_titanic:0.1

print('Приложение по определению вероятности запущено')
features_all = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
features_all_wo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

filepath_model = r'model/model_tk.pkl'
loaded_model = pickle.load(open(filepath_model, 'rb'))

#Зададим функцию предсказания
def prediction(p_Pclass, p_Sex, p_Age, p_SibSp, p_Parch, p_Fare, p_Embarked):
    y_predict = loaded_model.predict(pd.DataFrame([[p_Pclass, p_Sex, p_Age, p_SibSp, p_Parch, p_Fare, p_Embarked]]))

    return y_predict
# сделаем предсказание для пассажира
st.title('Приложение для определения вероятности выжить на Титанике')
st.image('pic.jpeg')
st.header('Для определение вероятности заполните характеристики пассажира :')

# Input text
p_Pclass = st.selectbox('Класс каюты пассажира (1-высший, 2- средняя палуба, 3 - у ватерлинии):', [1, 2, 3])
p_Sex = st.selectbox('Пол пассажира 0 - мужской, 1- женский:', [0, 1])
p_Age = st.number_input('Возраст пассажира в годах :', min_value=0, max_value=100, value=1)
p_SibSp = st.number_input('Число братьев, сестер или супругов на борту у пассажира:', min_value=0, max_value=20, value=1)
p_Parch = st.number_input('Число родителей или детей, с которыми путешествовал каждый пассажир:', min_value=0, max_value=20, value=1)
p_Fare = st.selectbox('Тариф пассажира в фунтах (Low - 7, Mid -15 , High_Mid - 31, High - 120)', [7, 15, 31, 120])
p_Embarked = st.selectbox('Порт посадки пассажира (1- Саутгемптон, 2 - Шербург и 3 - Куинстаун)', [1, 2, 3])

if st.button('Определить вероятность'):
    p = prediction(p_Pclass, p_Sex, p_Age, p_SibSp, p_Parch, p_Fare, p_Embarked)
    p_str = 'Вероятность, что пассажир выживет - ' + str(round(p[0], 2)*100) + '%'
    st.success(p_str)

print('Модель протестирована')

