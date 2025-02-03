import os
import joblib
import numpy as np
import streamlit as st
import gdown
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2

# URL файла на Google Drive
url = 'https://drive.google.com/uc?id=1AyPDoibUsYhx1CnFkFouPh_fIy0pXpB5'

# Локальный путь для сохранения файла
model_path = 'best_model_rf.joblib'

# Загрузка файла с Google Drive
#@st.cache(allow_output_mutation=True)
@st.cache_resource

def load_model_from_drive():
    gdown.download(url, model_path, quiet=False)
    model = joblib.load(model_path)
    return model

# Загрузка модели
model = load_model_from_drive()

# Функция для предварительной обработки изображения
def preprocess_image(image):
    try:
        #"""
        #Предварительная обработка нарисованного изображения для соответствия формату набора данных MNIST.
        #- Преобразование в градации серого.
        #- Изменение размера до 28x28 пикселей.
        #- Инвертирование цветов (MNIST использует белые цифры на черном фоне).
        #- Нормализация значений пикселей в диапазон [0, 1].
        #- Применение порогового значения для бинаризации изображения.
        #- Удаление шума с использованием морфологических операций.
        #- Преобразование изображения в одномерный массив из 784 элементов.
        #"""
        # Преобразование в градации серого и изменение размера
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image)
        
        # Инвертирование цветов (MNIST использует белые цифры на черном фоне)
        image_array = 255 - image_array
        
        # Применение размытия Гаусса для уменьшения шума
        image_array = cv2.GaussianBlur(image_array, (3, 3), 0)
        
        # Применение адаптивного порогового значения для бинаризации изображения
        image_array = cv2.adaptiveThreshold(
            image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Нормализация значений пикселей в диапазон [0, 1]
        image_array = image_array / 255.0
        
        return image_array
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {e}")
        return None

# Заголовок приложения Streamlit
st.title("Digit Recognition with MNIST")

# Создание трех колонок: левая (холст для рисования), средняя (вид модели), правая (результаты)
col1, col2, col3 = st.columns([1, 1, 1])  # Три колонки равной ширины

# Левая колонка: Холст для рисования
with col1:
    st.write("### 1. Нарисуйте цифру")
    
    # Создание компонента холста
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки (оранжевый с прозрачностью)
        stroke_width=25,  # Толщина линии
        stroke_color="#FFFFFF",  # Цвет линии (белый)
        background_color="#000000",  # Цвет фона (черный)
        update_streamlit=True,  # Обновление Streamlit при каждом изменении
        height=280,  # Высота холста
        width=280,  # Ширина холста
        drawing_mode="freedraw",  # Свободное рисование
        key="canvas",
    )

# Средняя колонка: То, что видит модель
with col2:
    if canvas_result.image_data is not None:
        st.write("### 2. То, что видит модель")
        #st.write("Вот как модель обрабатывает ваш рисунок:")
        
        # Преобразование изображения холста в формат PIL
        drawn_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        # Предварительная обработка изображения
        image_array = preprocess_image(drawn_image)
        
        if image_array is not None:
            # Отображение предварительно обработанного изображения (то, что видит модель)
            fig, ax = plt.subplots()
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')  # Скрыть оси
            st.pyplot(fig)

# Правая колонка: Результаты предсказаний и матрица путаницы
with col3:
    if canvas_result.image_data is not None and image_array is not None:
        st.write("### 3. Результаты предсказания")
        
        try:
            # Предсказание с использованием модели
            st.write("Выполнение предсказания...")
            prediction = model.predict(image_array.reshape(1, -1))
            prediction_proba = model.predict_proba(image_array.reshape(1, -1))  # Получение вероятностных оценок
            confidence = np.max(prediction_proba) * 100  # Расчет процента уверенности

            # Отображение результатов предсказания
            st.write(f"**Предсказанная цифра:** {prediction[0]}")
            st.write(f"**Уверенность предсказания:** {confidence:.2f}%")

            # Отображение распределения вероятностей для каждого класса
            st.write("**Распределение вероятностей:**")
            fig, ax = plt.subplots()
            classes = np.arange(10)  # Цифры от 0 до 9
            ax.bar(classes, prediction_proba[0], color='skyblue')
            ax.set_xlabel("Цифра")
            ax.set_ylabel("Вероятность")
            ax.set_xticks(classes)
            ax.set_title("Вероятность для каждой цифры")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
