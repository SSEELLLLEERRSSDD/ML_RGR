
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import rand_score
import pickle


page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)

def page_info():
    st.title("Разработка Web-приложения для инференса моделей ML и анализа данных")
    st.header("Автор")
    st.write("ФИО: Николаев Даниил Дмитриевич")
    st.write("Группа: МО-221")
    st.image("my_photo.jpg")

def page_datasetinfo():
    st.title("Информация о наборе данных")
    st.header("Тематика датасета")
    st.write("Дожди в Австралии")
    st.header("Описание признаков")
    st.write("- id: Идентификатор строки")
    st.write("- Date: Дата наблюдения")
    st.write("- Location: Общее название места расположения метеостанции")
    st.write("- MinTemp: Минимальная температура")
    st.write("- MaxTemp: Максимальная температура")
    st.write("- Rainfall: Количество осадков, выпавших за сутки в мм")
    st.write("- Evaporation: Испарение")
    st.write("- Sunshine: Количество часов яркого солнечного света в сутках")
    st.write("- WindGustDir: Направление самого сильного порыва ветра за сутки до полуночи")
    st.write("- WindGustSpeed: Скорость самого сильного порыва ветра за сутки до полуночи.")
    st.write("- WindDir9am: Направление ветра в 9 часов утра")
    st.write("- WindDir3pm: Направление ветра в 3 часа дня")
    st.write("- WindSpeed9am: Скорость ветра в 9 часов утра")
    st.write("- WindSpeed3pm: Скорость ветра в 3 часа дня")
    st.write("- Humidity9am: Влажность в 9 часов утра")
    st.write("- Humidity3pm: Влажность в 3 дня")
    st.write("- Pressure9am: Давление в 9 часов утра")
    st.write("- Pressure3pm: Давление в 3 часа дня")
    st.write("- Cloud9am: Погода в 9 часов утра")
    st.write("- Cloud3pm: Погода в 3 часа дня")
    st.write("- Temp9am: Погода в 9 часов утра")
    st.write("- Temp3pm: Температура в 3 часа дня")
    st.write("- RainToday: Был ли дождь сегодня")
    st.write("- RainTomorrow: Будет ли дождь завтра")
    st.header("Предобработка данных")
    st.write("В датасете необходимо предсказать будет ли дождь завтра. Будет дождь или не будет показателем 1 или 0.")
    st.write("1 - дождь будет")
    st.write("0 - дождя не будет")
    st.write("В датасете присутствовали категориальные признаки, так что было применено One-hot кодирование")
    st.write("Были удалены дубликаты")
    st.write("Было проведено EDA и удалены выбросы")
    st.write("Числовые признаки были масштабированны. Был устранен дисбаланс классов.")

def page_visualization():
    df = pd.read_csv('Weather_data.csv')

    st.title("Датасет Дожди в Австралии")

    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))
    selected_cols = ['RainTomorrow', 'RainToday', 'WindGustSpeed','Humidity9am','Temp9am', "Rainfall"]
    selected_df = df[selected_cols]
    sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)

    st.header("Гистограммы")

    columns = ['WindGustSpeed','Humidity9am','Temp9am']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df.sample(5000)[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)


    st.header("Боксплот")
    outlier = df[columns]
    Q1 = outlier.quantile(0.25)
    Q3 = outlier.quantile(0.75)
    IQR = Q3-Q1
    data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]


    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data_filtered[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)



    st.header("Графики целевого признака и наиболее коррелирующих с ним признаков")

    columns=[('RainTomorrow', 'RainToday'),
            ('RainTomorrow', 'Humidity9am'),
            ('RainTomorrow', 'WindGustSpeed'),
            ('RainTomorrow', 'Temp9am')]
    for col in columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(x=df[col[0]], y=df[col[1]])
        plt.xlabel(col[0])
        plt.ylabel(col[1])
        plt.title(f'{col[1]}')
        st.pyplot(plt)


def page_ml_prediction():
    with open('OneHotEncoder.pkl', 'rb') as file: 
        ColumnTransform = pickle.load(file)

    uploaded_file = st.file_uploader("Выберите файл датасета")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Загруженный датасет:", df)

    else:
        st.write("Датасет Дождь в Австралии")
        df = pd.read_csv('weatherAUS.csv')

    df.dropna(inplace=True,ignore_index=True)
    f = lambda x : str(x)[5:7]
    df['Date'] = df['Date'].transform(f)
    df['Date'] = df['Date'].astype(int)
    f = lambda x : 0 if (x == "No") else 1
    df['RainToday'] = df['RainToday'].transform(f)
    df['RainToday'] = df['RainToday'].astype(int)

    df['RainTomorrow'] = df['RainTomorrow'].transform(f)
    df['RainTomorrow'] = df['RainTomorrow'].astype(int)
    encoded_features = ColumnTransform.transform(df)
    data1=df.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)
    dataclass = pd.concat([
        data1,
        encoded_features
    ], axis=1)
    x_class=dataclass.drop(['RainTomorrow'],axis=1)
    y_class=dataclass['RainTomorrow']



    button_clicked_metrics = st.button("Расчитать точность моделей на датасете")

    if button_clicked_metrics:
        with open('SVC.plk', 'rb') as file:
            svc_model = pickle.load(file)

        with open('Bagging.plk', 'rb') as file:
            bagging_model = pickle.load(file)

        with open('GradientBoosting.plk', 'rb') as file:
            gradient_model = pickle.load(file)

        with open('Stacking_model.plk', 'rb') as file:
            stacking_model = pickle.load(file)

        from tensorflow.keras.models import load_model
        nn_model = load_model('NN.h5')


        st.header("SVC:")
        svc_pred = svc_model.predict(x_class)
        st.write('Accuracy: ',f"{accuracy_score(y_class, svc_pred)}")


        st.header("bagging:")
        bagging_pred = bagging_model.predict(x_class)
        st.write('Accuracy: ',f"{accuracy_score(y_class, bagging_pred)}")

        st.header("gradient:")
        gradient_pred = gradient_model.predict(x_class)
        st.write('Accuracy: ',f"{accuracy_score(y_class, gradient_pred)}")

        st.header("Perceptron:")
        nn_pred = [np.argmax(pred) for pred in nn_model.predict(x_class, verbose=None)]
        st.write('Accuracy: ',f"{accuracy_score(y_class, nn_pred)}")

        st.header("Stacking:")
        stacking_pred = stacking_model.predict(x_class)
        st.write('Accuracy: ',f"{accuracy_score(y_class, stacking_pred)}")


    st.title("Получить предсказание дождя.")

    st.header("Date")
    Date = st.number_input("Месяц:", value=1, min_value=1, max_value=12)

    st.header("Location")
    locations = ['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney',
        'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Canberra', 'Sale',
        'MelbourneAirport', 'Melbourne', 'Mildura', 'Portland', 'Watsonia',
        'Brisbane', 'Cairns', 'Townsville', 'MountGambier', 'Nuriootpa',
        'Woomera', 'PerthAirport', 'Perth', 'Hobart', 'AliceSprings',
        'Darwin']
    Location = st.selectbox("Город", locations)

    st.header("MinTemp")
    MinTemp = st.number_input("Число:", value=20.0)

    st.header("MaxTemp")
    MaxTemp = st.number_input("Число:", value=40.0)

    st.header("Rainfall")
    Rainfall = st.number_input("Число:", value=200, min_value=0, max_value=999999)

    st.header("Evaporation")
    Evaporation = st.number_input("Число:", value=12.8)

    st.header("Sunshine")
    Sunshine = st.number_input("Число:", value=13.2, min_value=0, max_value=999999)

    st.header("WindGustDir")
    dirs=['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W',
        'WSW', 'NNW', 'ESE', 'SSE', 'NW']
    WindGustDir = st.selectbox("Направление", dirs)

    st.header("WindGustSpeed")
    WindGustSpeed = st.number_input("Число:", value=30, min_value=0, max_value=999999)

    st.header("WindDir9am")
    dirs2=['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW',
        'E', 'NNW', 'ESE', 'WSW', 'SW']
    WindDir9am = st.selectbox("Направление", dirs2)

    st.header("WindDir3pm")
    dirs3=['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE',
        'NW', 'E', 'ESE', 'NE', 'SSW', 'W']
    WindDir3pm = st.selectbox("Направление", dirs3)

    st.header("WindSpeed9am")
    WindSpeed9am = st.number_input("Число:", value=11, min_value=0, max_value=999999)

    st.header("WindSpeed3pm")
    WindSpeed3pm = st.number_input("Число:", value=7, min_value=0, max_value=999999)

    st.header("Humidity9am")
    Humidity9am = st.number_input("Число:", value=27, min_value=0, max_value=999999)

    st.header("Humidity3pm")
    Humidity3pm = st.number_input("Число:", value=9, min_value=0, max_value=999999)

    st.header("Pressure9am")
    Pressure9am = st.number_input("Число:", value=1012.6, min_value=0, max_value=999999)

    st.header("Pressure3pm")
    Pressure3pm = st.number_input("Число:", value=1010.1, min_value=0, max_value=999999)

    st.header("Cloud9am")
    Cloud9am = st.number_input("Число:", value=0.1, min_value=0, max_value=999999)

    st.header("Cloud3pm")
    Cloud3pm = st.number_input("Число:", value=1, min_value=0, max_value=999999)

    st.header("Temp9am")
    Temp9am = st.number_input("Число:", value=29.8, min_value=0, max_value=999999)

    st.header("Temp3pm")
    Temp3pm = st.number_input("Число:", value=36.4, min_value=0, max_value=999999)

    st.header("RainToday")
    RainToday = st.number_input("Число:", value=0, min_value=0, max_value=1)

    data = pd.DataFrame({'Date': [Date],
                        'Location': [Location],
                        'MinTemp': [MinTemp],
                        'MaxTemp': [MaxTemp],
                        'Rainfall': [Rainfall],
                        'Evaporation': [Evaporation],
                        'Sunshine': [Sunshine],
                        'WindGustDir': [WindGustDir],
                        'WindGustSpeed': [WindGustSpeed],
                        'WindDir9am': [WindDir9am],
                        'WindDir3pm': [WindDir3pm],
                        'WindSpeed9am': [WindSpeed9am],
                        'WindSpeed3pm': [WindSpeed3pm],
                        'Humidity9am': [Humidity9am],    
                        'Humidity3pm': [Humidity3pm],   
                        'Pressure9am': [Pressure9am],   
                        'Pressure3pm': [Pressure3pm],   
                        'Cloud9am': [Cloud9am],   
                        'Cloud3pm': [Cloud3pm],   
                        'Temp9am': [Temp9am],   
                        'Temp3pm': [Temp3pm],   
                        'RainToday': [RainToday],  
                        'RainTomorrow': [0]       
                        })


    encoded_features = ColumnTransform.transform(data)
    data1=data.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)
    df = pd.concat([
        data1,
        encoded_features
    ], axis=1)
    df=df.drop(['RainTomorrow'],axis=1)




    button_clicked = st.button("Предсказать")

    if button_clicked:
        with open('SVC.plk', 'rb') as file:
            svc_model = pickle.load(file)

        with open('Bagging.plk', 'rb') as file:
            bagging_model = pickle.load(file)

        with open('GradientBoosting.plk', 'rb') as file:
            gradient_model = pickle.load(file)

        with open('Stacking_model.plk', 'rb') as file:
            stacking_model = pickle.load(file)
        from tensorflow.keras.models import load_model
        nn_model = load_model('NN.h5')


        st.header("SVC:")
        pred =[]
        svc_pred = svc_model.predict(df)[0]
        pred.append(int(svc_pred))
        st.write(f"{svc_pred}")


        st.header("bagging:")
        bagging_pred = bagging_model.predict(df)[0]
        pred.append(int(bagging_pred))
        st.write(f"{bagging_pred}")

        st.header("gradient:")
        gradient_pred = gradient_model.predict(df)[0]
        pred.append(int(gradient_pred))
        st.write(f"{gradient_pred}")

        st.header("Perceptron:")
        nn_pred = round(nn_model.predict(df)[0][0])
        pred.append(nn_pred)
        st.write(f"{nn_pred}")

        st.header("Stacking:")
        stacking_pred = stacking_model.predict(df)[0]
        pred.append(int(stacking_pred))
        st.write(f"{stacking_model.predict(df)[0]}")


if page == "Информация о разработчике":
    page_info()
elif page == "Информация о наборе данных":
    page_datasetinfo()
elif page == "Визуализации данных":
    page_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()
