import pickle
import numpy as np
import json
import re
from tqdm import tqdm
from uuid import uuid4
import os
import torch
from torch.utils.data import Dataset, DataLoader
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from encoder import EncoderCNN
import torch.nn as nn     

items = pickle.load(open('../dataset/nutritional_item.pkl', 'rb'))
image_dir = '../dataset/Recipe1M'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using ', device)
resnet_features = []
transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

transf_list = []
transf_list.append(transforms.Resize(256))
transf_list.append(transforms.CenterCrop(224))
transform = transforms.Compose(transf_list)

model = EncoderCNN(128, 0.3, 'resnet50').to(device)
#encoder_image = encoder_image.cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids = [0,1])

for item in tqdm(items[30000:]):
    for j in item['images']:
        path = os.path.join(image_dir, item['partition'], j[0], j[1], j[2], j[3], j)
        if os.path.exists(path):
            dict_resnet_features = {}
            image = Image.open(path).convert('RGB')

            image_transf = transform(image)
            image_tensor = to_input_transf(
                image_transf).unsqueeze(0).to(device)

            feature = model.forward(image_tensor)

            dict_resnet_features['feature'] = feature
            dict_resnet_features['image'] = j
            dict_resnet_features['ingredients'] = item['ingredients']
            dict_resnet_features['lights'] = item['lights']
            resnet_features.append(dict_resnet_features)
            del image_tensor
file = open('../dataset/nutritional_training_128_3.pickle', 'wb')
pickle.dump(resnet_features, file)
file.close()
