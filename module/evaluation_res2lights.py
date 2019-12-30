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
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512 * 49, 3000)  # 6*6 from image dimension
        self.fc2 = nn.Linear(3000,500)
        self.fc3 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

device_id = 0
device = 'cuda:' + str(device_id)
torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = torch.load('./model_res2lights.pkl')

items = pickle.load(open('../dataset/nutritional_training_all.pickle', 'rb'))
features = []
fats = []
light_dict = {
    'green': 0,
    'orange': 1,
    'red': 2
}
for item in items[20000:]:
    features.append(item['feature'].view(-1))
    fats.append(light_dict[item['lights']['fat']])
print('loaded')

total = len(fats)

del items
features = torch.stack(features).to(device)
features = Variable(features, requires_grad=True)
fats = torch.FloatTensor(fats).to(device=device, dtype=torch.int64)
torch_dataset = Data.TensorDataset(features,fats)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=50,      # mini batch size
    shuffle=False,              # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    drop_last=False
)

correct = 0

for image, label in tqdm(loader):
    pred = model(image)
    value, index = torch.max(pred,1)
    #print(label,pred)
    error = index - label
    #print(label, value, error)
    for i in error:
        if i == 0:
            correct += 1
    #print(index, label)
    #if pred == label:
    #    correct += 1
print(correct, total, 1.0 * correct/total )
