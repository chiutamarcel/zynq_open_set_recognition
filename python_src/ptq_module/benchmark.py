#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WORK IN PROGRESS
"""

import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import qtensor as ql

torch.manual_seed(42)

INPUT_DIM = 784
OUTPUT_DIM = 10
INPUT_MAX = 1
INPUT_MIN = 0
        
def train_model(model, criterion, optimizer, train_loader):
    for epoch in range(5):  # Number of epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

def eval_model(model, test_loader):
    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
def eval_quant_inf(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            qx, qx_scale = ql.quantize(inputs, min_val=-1, max_val=1)
            outputs =  ql.inference(qx, model)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

#%%
load_model = True
save_model = False

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL.Image to torch.Tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    transforms.Lambda(lambda x: torch.flatten(x))
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

#%%
# Get one image and its label from the train loader
data_iter = iter(train_loader)
single_image, single_label = next(data_iter)
single_image = single_image[0]
single_label = single_label[0]

single_image

#%%
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, kernel_size=(3, 3)),
    torch.nn.Conv2d(4, 8, kernel_size=(3, 3)),
    # torch.nn.Linear(INPUT_DIM, 256),
    # torch.nn.Linear(256, 32),
    # torch.nn.Linear(32, 10),
    torch.nn.ReLU()
)
# TODO: add convolutional layers to model and compare with quantized model

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if load_model:
    model.load_state_dict(torch.load("model.pth"))
else:
    train_model(model, criterion, optimizer, train_loader)
    eval_model(model, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "model.pth")

#%% Print results on a random image, just to see what they look like

ql.inference(single_image, model)

#%%
print("[Quantized Model]")
eval_quant_inf(model, test_loader)

print("[Original Model]")
eval_model(model, test_loader)

#%%

N = 1
IN_CHANNELS = 1
OUT_CHANNELS = 1

x = torch.rand(N, IN_CHANNELS, 64, 64)
qx, qx_scale = ql.quantize(x)
layer = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, 3, bias=True)

y1 = layer(x).detach()
y2, y2_scale = ql.conv2d(x, layer)
# y2 = torch.zeros_like(y1)

# print("y1:", y1)
# print("y2:", y2)

#%%

# Check if elements are close ( they are float, so they can't be exactly equal )
assert torch.allclose(y1, y2, atol=1e-01), "Incorrect output!"

print("Output is correct.")
