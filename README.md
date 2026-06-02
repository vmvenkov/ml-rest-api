# ML REST API with Docker
Описание
Този проект представлява REST API система за прогнозиране чрез модел за машинно обучение.

Използван е Iris Dataset и алгоритъм Random Forest Classifier от библиотеката scikit-learn.

Системата предоставя REST метод за прогнозиране на вида ирис въз основа на четири входни характеристики:
* sepal_length
* sepal_width
* petal_length
* petal_width
  
API валидира входните данни и връща подходящи съобщения за грешки при невалидни заявки или проблеми при зареждането на модела.
Използвани технологии
* Python 3.11
* FastAPI
* Scikit-learn
* Joblib
* Docker
* Pytest

Структура на проекта

* app/
    * main.py
    * model.py
    * train_model.py
* tests/
    * test_api.py
* Dockerfile
* requirements.txt
* README.md
* .gitignore


Създаване на Docker Image

Изпълнете:

docker build -t ml-rest-api .

Стартиране на контейнер

Изпълнете:

docker run -p 8000:8000 ml-rest-api

След стартиране приложението ще бъде достъпно на:

http://localhost:8000

Swagger документация:

http://localhost:8000/docs

Пример за прогнозиране
POST заявка към:
http://localhost:8000/predict

Примерен JSON:
{
  "sepal_length": 5.1,
  
  "sepal_width": 3.5,
  
  "petal_length": 1.4,
  
  "petal_width": 0.2
}

Примерен отговор:
{
  "prediction": 0,
  
  "class_name": "setosa"
}

Обработка на грешки
* HTTP 422 – невалидни входни данни
* HTTP 500 – проблем при зареждане на модела или при прогнозиране
  
Стартиране на тестове

pytest
