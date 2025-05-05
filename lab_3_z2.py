import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Загрузка данных из CSV-файла
df = pd.read_csv('data.csv')

# Просмотр первых строк для понимания структуры данных
print(df.head())

# Разделение данных на признаки (X) и метки (y)
# Предполагается, что столбцы 0, 1, 2, 3 содержат параметры, а последний столбец — вид растения
X = df.iloc[:, :4].values  # Первые 4 столбца
y = df.iloc[:, 4].values   # Последний столбец (вид растения)

# Преобразование меток в числовой формат
# Например: "Iris-setosa" -> 0, "Iris-versicolor" -> 1
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1}
y = np.array([label_map[label] for label in y])

# Преобразование данных в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Создание полносвязного слоя (нейронной сети)
# Входной размер: 4 (количество параметров), выходной размер: 3 (количество классов)
model = nn.Linear(4, 1)

# Определение функции потерь и оптимизатора
loss_fn = nn.MSELoss()  # Подходит для задач классификации
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Цикл обучения
num_epochs = 100
for epoch in range(num_epochs):
    # Прямой проход (предсказание)
    outputs = model(X_tensor)
    # outputs = outputs>0.5
    # Вычисление ошибки
    loss = loss_fn(outputs.to(torch.float32), y_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуление градиентов
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление весов
    
    # Вывод ошибки каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Ошибка: {loss.item():.4f}')

# Тестирование модели
with torch.no_grad():  # Отключаем вычисление градиентов
    predictions = model(X_tensor)
    _, predicted_classes = torch.max(predictions, 1)  # Получаем предсказанные классы

# Сравнение результатов с эталонным тензором t
# Предположим, что t уже задан и содержит правильные метки
t = torch.tensor(y, dtype=torch.long)  # Эталонный тензор


# Вывод результатов
print("Предсказанные классы:", predicted_classes)
print("Эталонные метки:", t)