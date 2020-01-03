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

        self.fc1 = nn.Linear(512 * 49, 300)  
        self.fc2 = nn.Linear(300,3)
        #self.fc3 = nn.Linear(500, 100)
        #self.fc4 = nn.Linear(100,3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def data_parallel(model, input, device_ids, output_device=None):
    if not device_ids:
        return model(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(model, device_ids)
    print(f"replicas:{replicas}")
    
    inputs = nn.parallel.scatter(input, device_ids)
    print(f"inputs:{type(inputs)}")
    for i in range(len(inputs)):
        print(f"input {i}:{inputs[i].shape}")
        
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    print(f"outputs:{type(outputs)}")
    for i in range(len(outputs)):
        print(f"output {i}:{outputs[i].shape}")
        
    result = nn.parallel.gather(outputs, output_device)
    return result


torch.cuda.empty_cache()
# device_id = 0
# device = 'cuda:' + str(device_id)
# torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# print('cuda',torch.cuda.is_available())

BATCH_SIZE = 128
epochs = 10
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

#features = torch.stack(features).to(device)
features = torch.stack(features)
fats = torch.FloatTensor(fats)
torch_dataset = Data.TensorDataset(features,fats)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,              # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    drop_last=False
)

model = Model()
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
        output = data_parallel(model.cuda(),feature.cuda(), [0,1])
        # output = model(feature)
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
        print(f"Training loss: {running_loss/len(loader)}")
        print('Accuracy: ',correct/totalcount)

torch.save(model, 'model_res2lights.pkl')
