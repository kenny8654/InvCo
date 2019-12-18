import pickle
import pandas as pd
import numpy as np
import json
import re
from tqdm
from uuid import uuid4
import os
import torch
from torch.utils.data import Dataset, DataLoader
import urllib
from PIL import Image
from torchvision import transforms


items = pickle.load(open('../dataset/nutritional_filtered.pickle', 'rb'))
image_dir = '../dataset/Recipe1M/zip/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)
resnet_features = []
transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

for item in tqdm(items):
    for j in item['images']:
        path = os.path.join(
            image_dir, item['partition'], j[0], j[1], j[2], j[3], j)
        if os.path.exists(path):
            dict_resnet_features = {}
            image = Image.open(path).convert('RGB')

            transf_list = []
            transf_list.append(transforms.Resize(256))
            transf_list.append(transforms.CenterCrop(224))
            transform = transforms.Compose(transf_list)

            image_transf = transform(image)
            image_tensor = to_input_transf(
                image_transf).unsqueeze(0).to(device)
            encoder_image = EncoderCNN(512, 0.3, 'resnet101')
            feature = encoder_image.forward(image_tensor)

            dict_resnet_features['feature'] = feature
            dict_resnet_features['image'] = j
            dict_resnet_features['ingredients'] = item['ingredients']
            dict_resnet_features['lights'] = item['lights']
            resnet_features.append(dict_resnet_features)
