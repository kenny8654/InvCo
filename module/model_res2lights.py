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

        self.fc1 = nn.Linear(128 * 36, 1000)  
        self.fc2 = nn.Linear(1000,3)
        #self.fc3 = nn.Linear(500, 100)
        #self.fc4 = nn.Linear(100,3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


torch.cuda.empty_cache()
device_id = 0
device = 'cuda:' + str(device_id)
torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('cuda',torch.cuda.is_available())

BATCH_SIZE = 128
epochs = 10
items = pickle.load(open('../dataset/nutritional_resnet_128.pkl', 'rb'))
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

#del items
features = torch.stack(features).to(device)
#features = torch.stack(features).to('cpu')
features = Variable(features, requires_grad=True)
#features = features.requires_grad_()
print(items[0]['feature'].size())
#fats = torch.FloatTensor(fats).to(device=device, dtype=torch.int64)
fats = torch.FloatTensor(fats)
#fats = torch.stack(fats).to(device)
print(features, fats)
torch_dataset = Data.TensorDataset(features,fats)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,              # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    drop_last=False
)
model = Model()
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#    model = nn.DataParallel(model, device_ids = [0,1])
#    features = nn.DataParallel(features, device_ids = [0,1])
model.to(device)
#model = Model().to(device)
#model = nn.DataParallel(model)
print(model)
#criterion = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
criterion = nn.CrossEntropyLoss()
#criterion = Variable(criterion, requires_grad = True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)

for e in range(epochs):
    running_loss = 0
    correct = 0
    totalcount = 0
    for feature, labels in tqdm(loader):
        optimizer.zero_grad()  # 避免上一個batch的gradient累積
        #with torch.no_grad():
        #feature = feature.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        output = model(feature.to(device))
        #print('output : %s , labels : %s '%(output,labels))
        #_, predicted = torch.max(output, 1)
        #print('predicted : %s '%(predicted))
        loss = criterion(output, labels)
        #_, predicted = torch.max(outputs, 1)
        #loss.item()
        #print('ouput : %s , predicted : %s , labels : %s '%('ouput','predicted','labels'))
        loss.backward()
        #loss.item()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        test = predicted == labels.long()
        #print('predicted : ',predicted)
        #print('labels : ',labels)
        #print('test : ',test)
        correct += test.sum().item()
        totalcount += len(test)
        #print('correct : ',correct)
        #print('totalcount : ',totalcount)
        del feature
        del labels

    else:
        print(f"Training loss: {running_loss/len(loader)}")
        print('Accuracy: ',correct/totalcount)

torch.save(model, 'model_res2lights.pkl')
