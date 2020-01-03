import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
import pickle
from uuid import uuid4
import os
import urllib
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from encoder import EncoderCNN


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512 * 49, 300)  
        self.fc2 = nn.Linear(300,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

torch.cuda.empty_cache()
device_id = 0
device = 'cuda:' + str(device_id)
torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('cuda',torch.cuda.is_available())


items = pickle.load(open('../dataset/nutritional_training_all.pickle', 'rb'))
features = []
fats = []

#torch.FloatTensor([1,0,0])
light_dict = {
    'green': 0,
    'orange': 1,
    'red': 2
}
for item in items:
    features.append(item['feature'].view(-1))
    fats.append(light_dict[item['lights']['fat']])
print('loaded')

del items
#features = torch.stack(features).to(device)
features = torch.stack(features).to('cpu')
features = Variable(features, requires_grad=True)
fats = torch.FloatTensor(fats)
#fats = torch.stack(fats).to(device)
# print(features, fats)
torch_dataset = Data.TensorDataset(features,fats)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,              # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    drop_last=False
)

model = Model().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)

for e in range(epochs):
    running_loss = 0
    correct = 0
    totalcount = 0
    for feature, labels in tqdm(loader):
        optimizer.zero_grad()  # 避免上一個batch的gradient累積
        feature = feature.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        output = model(feature)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        test = predicted == labels.long()
        # print('predicted : ',predicted)
        # print('labels : ',labels)
        # print('test : ',test)
        correct += test.sum().item()
        totalcount += len(test)
        # print('correct : ',correct)
        # print('totalcount : ',totalcount)
        del feature
        del labels

    else:
        print(f"Training loss: {running_loss/len(loader)}")
        print('Accuracy: ',correct/totalcount)

torch.save(model, 'model_res2lights.pkl')
