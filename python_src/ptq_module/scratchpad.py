import torch

N = 10
x = torch.rand(N)
layer = torch.nn.Softmax()

y = layer(x)

print(f"x:\n{x}")
print(f"y:\n{y}")