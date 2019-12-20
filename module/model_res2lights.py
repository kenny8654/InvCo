import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable

learning_rate = 0.0001
BATCH_SIZE = 64
epochs = 10
res_batch_size = 1000
items = pickle.load(open('../dataset/nutritional_training_all.pickle', 'rb'))

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(512 * 49, 3000)  
        self.fc2 = nn.Linear(3000,3)
        #self.fc3 = nn.Linear(500, 100)
        #self.fc4 = nn.Linear(100,3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def res_generator(items,batch_size = res_batch_size):
    while True:
        for index in range(0,len(items),batch_size):
            yield items[index: index+batch_size] 

torch.cuda.empty_cache()
device_id = 0
device = 'cuda:' + str(device_id)
torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('cuda is available : ',torch.cuda.is_available())

#torch.FloatTensor([1,0,0])
light_dict = {
    'green': 0,
    'orange': 1,
    'red': 2
}

def dataLoader(items):
    features = []
    fats = []
    for item in tqdm(items):
        features.append(item['feature'].view(-1))
        fats.append(light_dict[item['lights']['fat']])
    print('loaded')
    
    # del items
    features = torch.stack(features).to(device)
    features = Variable(features, requires_grad=True)
    #features = features.requires_grad_()

    fats = torch.FloatTensor(fats).to(device=device, dtype=torch.int64)
    #fats = torch.stack(fats).to(device)
    print(features, fats)
    torch_dataset = Data.TensorDataset(features,fats)
    loader = Data.DataLoader(
        dataset=torch_dataset,     
        batch_size=BATCH_SIZE,      
        shuffle=True,              
        num_workers=0,              
        drop_last=False
    )
    return loader

model = Model().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)

generator = res_generator(items)
for e in range(epochs):
    running_loss = 0
    correct = 0
    totalcount = 0

    for i in range(len(items)//res_batch_size):
        item = next(generator)
        print('item len : ',len(item))
        loader = dataLoader(item)

        for feature, labels in tqdm(loader):
            optimizer.zero_grad()  # 避免上一個batch的gradient累積
            output = model(feature)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            test = predicted == labels.long()
            correct += test.sum().item()
            totalcount += len(test)
            print('predicted : ',predicted)
            print('labels : ',labels)
            print('test : ',test)
            print('correct : ',correct)
            print('totalcount : ',totalcount)

        else:
            print(f"Training loss: {running_loss/len(loader)}")
            print('Accuracy: ',correct/totalcount)
    torch.cuda.empty_cache()
torch.save(model, 'model_res2lights.pkl')
