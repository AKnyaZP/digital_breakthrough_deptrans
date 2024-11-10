# Используйте базовый образ Python
FROM python:3.10-slim

# Установите необходимые зависимости (если есть)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Скопируйте ваш Python файл в образ
COPY model.py
COPY test.py

# Задайте команду для запуска вашего Python файла
CMD ["python", "test.py"]
