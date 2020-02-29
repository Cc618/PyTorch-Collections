
import torch as T
from torch.autograd import Variable


# Hyper params
lr = 0.25
momentum = .25
n_data = 20
epochs = 100

# Dataset (normalized)
# y = 2 * x + 1
data_x = T.tensor([x / n_data for x in range(n_data)], dtype=T.float)
data_y = T.tensor([2 * (x / n_data) + 1 for x in range(n_data)], dtype=T.float)

# Weights
w = Variable(T.tensor([.2, .1], dtype=T.float), requires_grad=True)


def forward(x):
    global w

    return x * w[0] + w[1]


def loss(pred, y):
    global n_data

    return 1 / n_data * (pred - y) ** 2


def test():
    global data_x, data_y

    mse = 0
    for x, y in zip(data_x, data_y):
        pred = forward(x)
        mse += loss(pred, y)

    return mse.item()


print(f'Init mse : {test():.4f}')

# Train
for epoch in range(epochs):
    for x, y in zip(data_x, data_y):
        # Feed forward
        pred = forward(x)
        error = loss(pred, y)

        # Back prop
        error.backward()
        w.data -= lr * w.grad.data

        # Momentum
        w.grad.data *= momentum
        # Without momentum : w.grad.data.zero_()

print(f'After train mse : {test():.4f}')

print('y    pred')
# Evaluate
for x, y in zip(data_x, data_y):
    pred = x * w[0] + w[1]
    print(f'{y.item():.2f} {pred.item():.2f}')

print(f'Weights : {w[0].item():.2f} {w[1].item():.2f}')
