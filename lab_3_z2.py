import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Загрузка данных из CSV-файла
df = pd.read_csv('data.csv')

# Просмотр первых строк для понимания структуры данных
print("Первые 5 строк данных:")
print(df.head())

# Разделение данных на признаки (X) и метки (y)
X = df.iloc[:, :4].values  # Первые 4 столбца
y = df.iloc[:, 4].values   # Последний столбец (вид растения)

# Преобразование меток в числовой формат
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
y = np.array([label_map[label] for label in y])

# Вычисляем среднее и стандартное отклонение для каждого признака
mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std

# Разделение данных на обучающую и тестовую выборки вручную (80% - обучение, 20% - тест)
np.random.seed(42)  # Для воспроизводимости
indices = np.random.permutation(len(X_normalized))
split_index = int(0.8 * len(X_normalized))  # 80% для обучения
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train, X_test = X_normalized[train_indices], X_normalized[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Скрытый слой с 10 нейронами
        self.relu = nn.ReLU()        # Функция активации ReLU
        self.fc2 = nn.Linear(10, 3)  # Выходной слой с 3 нейронами (для 3 классов)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()

# Определение функции потерь и оптимизатора
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Цикл обучения
num_epochs = 100
for epoch in range(num_epochs):
    # Прямой проход (предсказание)
    outputs = model(X_train_tensor)
    
    # Вычисление ошибки
    loss = loss_fn(outputs, y_train_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуление градиентов
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление весов
    
    # Вывод ошибки каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Ошибка: {loss.item():.4f}')

# Тестирование модели
with torch.no_grad():  # Отключаем вычисление градиентов
    predictions = model(X_test_tensor)
    _, predicted_classes = torch.max(predictions, 1)  # Получаем предсказанные классы

# Оценка точности модели
accuracy = (predicted_classes == y_test_tensor).sum().item() / len(y_test_tensor)
print(f'\nТочность модели на тестовых данных: {accuracy * 100:.2f}%')

# Вывод результатов
print("\nПредсказанные классы:", predicted_classes)
print("Эталонные метки:", y_test_tensor)