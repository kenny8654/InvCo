import pickle
import numpy as np
import json
import re
from tqdm import tqdm
from uuid import uuid4
import os
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from encoder import EncoderCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512 * 49, 300)  
        self.fc2 = nn.Linear(300,3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def dataloader(items,batch_size=32):
    index = 0
    light_dict = {
        'green': 0,
        'orange': 1,
        'red': 2
    }
    resnet_features = []
    transf_list_batch = []
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    encoder_image = EncoderCNN(512, 0.3, 'resnet50')
    encoder_image = encoder_image.cuda()
    image_dir = '../dataset/Recipe1M/'
    while True:
        features = []
        labels = []
        for batch in range(0,batch_size):
            while True:
                for item in items[index]['images']:
                    print('item : ',item)
                    path = path = os.path.join(image_dir, items[index]['partition'], item[0], item[1], item[2], item[3], item)
                    if os.path.exists(path):
                        count = (index + 1)//len(items)
                        index = (index + 1)%len(items)
                        image = Image.open(path).convert('RGB')
                        image_transf = transform(image)
                        image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
                        feature = encoder_image.forward(image_tensor)
                        features.append(feature.view(-1))
                        labels.append(light_dict[item['lights']['fat']])
            yield features,labels,count

print(torch.cuda.is_available())
items = pickle.load(open('../dataset/nutritional_filtered.pickle', 'rb'))
device_id = 0
device = 'cuda:' + str(device_id)
torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('using ', device)
loader = dataloader(items)

epochs = 10
model = Model().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
torch.set_grad_enabled(True)

for e in tqdm(range(epochs)):
    torch.cuda.empty_cache()
    running_loss = 0
    correct = 0
    totalcount = 0
    while True:
        feature, label, count = next(loader)
        feature = torch.stack(feature).to(device)
        feature = Variable(feature, requires_grad=True)
        label = torch.FloatTensor(label).to(device=device, dtype=torch.int64)

        optimizer.zero_grad() 
        output = model(feature)
        loss = criterion(output, label)
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
        print('Accuracy: ',correct/totalcount)
        del feature
        del label
        if count > e:
            break

    print('epoch : ',e)
    print(f"Training loss: {running_loss/len(loader)}")
    print('Accuracy: ',correct/totalcount)

torch.save(model, 'model_res2lights.pkl')