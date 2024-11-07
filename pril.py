
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib  # для загрузки обученной модели

# Загрузка модели
model = joblib.load('random_forest_model.pkl')

# Настройка интерфейса
st.title("Прогноз стоимости подержанного автомобиля")

# Ввод данных пользователем
year = st.slider("Год выпуска", 1980, 2022, 2010)
km_driven = st.number_input("Пробег (в км)", min_value=0, value=50000)
fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
mileage = st.number_input("Пробег (км/л)", min_value=0.0, value=15.0)
engine = st.number_input("Объем двигателя (cc)", min_value=500, value=1000)
max_power = st.number_input("Максимальная мощность (bhp)", min_value=10, value=100)
seats = st.slider("Количество мест", 2, 8, 5)

# Подготовка данных для предсказания
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

# Предсказание цены продажи
prediction = model.predict(input_data)[0]
st.write(f"Предсказанная стоимость автомобиля: ${prediction:,.2f}")
