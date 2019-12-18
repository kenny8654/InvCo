import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
import pickle

# Torch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(512 * 49, 20)  # 6*6 from image dimension
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


BATCH_SIZE = 20
epochs = 10
items = pickle.load(open('../dataset/nutritional_training_all.pickle', 'rb'))
features = []
fats = []
light_dict = {
    'green': 0,
    'orange': 1,
    'red': 2
}
for item in items:
    features.append(item['feature'].view(-1))
    fats.append(light_dict[item['lights']['fat']])

# features = torch.FloatTensor(features)
torch_dataset = Data.TensorDataset(data_tensor=features, target_tensor=fats)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

model = Model()
print(model)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for e in range(epochs):
    running_loss = 0
    for feature, labels in loader:
        optimizer.zero_grad()  # 避免上一個batch的gradient累積
        output = model(feature)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(loader)}")
