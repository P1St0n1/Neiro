import torch
import random

# Задание 1
x = torch.zeros(1, 5, dtype=torch.int)
x = torch.randint(1, 10, (1, 5))
x = x.to(dtype=torch.float)
x.requires_grad=True
print("Исходный тензор:\n", x)
y = x**3
r = random.randint(1, 10)
print("Rand = ", r)
z = y*r
print("Обновленный тензор:\n", z)
exp_x = torch.exp(z)
res = exp_x.mean()
res.backward()
print("Производная exp(x) по x:\n", x.grad)