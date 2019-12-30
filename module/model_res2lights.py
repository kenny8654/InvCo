import numpy as np
import json
import re
from tqdm import tqdm
import pickle
import requests
# Torch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable

learning_rate = 0.00001 #0.0001
BATCH_SIZE = 64
epochs = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512 * 49, 3000)  
        self.fc2 = nn.Linear(3000,3)
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


items = pickle.load(open('nutritional_training_all.pickle', 'rb'))
light_dict = {
    'green': 0,
    'orange': 1,
    'red': 2
}

model = Model()
model = Model().to(device)
print(model)
#criterion = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
torch.set_grad_enabled(True)

for e in range(epochs):
    running_loss = 0
    correct = 0
    totalcount = 0
    item_count = 0
    item_in_gpu = 15000
    
    while True:
        if item_count + item_in_gpu >= len(items):
            item_in_gpu = len(items) - item_count

        features = []
        fats = []
        for item in items[item_count:item_count+item_in_gpu]:
            features.append(item['feature'].view(-1))
            fats.append(light_dict[item['lights']['sugars']])
        print('loaded')
        features = torch.stack(features)
        features = Variable(features, requires_grad=True)
        fats = torch.FloatTensor(fats)
        torch_dataset = Data.TensorDataset(features,fats)
        loader = Data.DataLoader(
            dataset=torch_dataset,     
            batch_size=BATCH_SIZE,      
            shuffle=True,              
            num_workers=0,              
            drop_last=False
        )

        for feature, labels in tqdm(loader):
            optimizer.zero_grad()  
            feature = feature.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            output = model(feature)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            test = predicted == labels.long()
            correct += test.sum().item()
            totalcount += len(test)
            
            del feature
            del labels

        else:
            # print("Training loss: {running_loss/len(loader)}")
            print('Epoch: ',e,', Accuracy: ',correct/totalcount)
            del loader
        item_count = item_count + item_in_gpu
        # print('epoch: ',e,'item_in_gpu: ',item_in_gpu,' item_count: ',item_count,' len(items): ',len(items))
        if item_in_gpu != 15000:
            break

torch.save(model, 'model_res2lights_sugars.pkl')
