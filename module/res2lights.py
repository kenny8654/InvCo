import os
import sys
import pickle
import time

# Torch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512 * 49, 3000)  
        self.fc2 = nn.Linear(3000,3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0), nn.Dropout2d(dropout))

    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.resnet(images)
        features = self.linear(raw_conv_feats)
        features = features.view(features.size(0), features.size(1), -1)

        return features


class Res2lights():

    def get_lights(self, image_name):
        start = time.time()
        image_dir = '/home/r8v10/git/InvCo/web/media/img/'
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        device = 'cpu'
        # torch.cuda.set_device(device)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_grad_enabled(True)

        model_fat = torch.load('/home/r8v10/git/InvCo/module/saved_model/model_res2lights_fat.pkl' ,map_location="cpu")
        model_salt = torch.load('/home/r8v10/git/InvCo/module/saved_model/model_res2lights_salt.pkl',map_location="cpu")
        model_sugars = torch.load('/home/r8v10/git/InvCo/module/saved_model/model_res2lights_sugars.pkl',map_location="cpu")
        model_saturates = torch.load('/home/r8v10/git/InvCo/module/saved_model/model_res2lights_saturates.pkl',map_location="cpu")
        print('model:',time.time() - start)

        light_dict = {'green': 0, 'orange': 1, 'red': 2}

        resnet_features = []
        transf_list_batch = []
        transf_list_batch.append(transforms.ToTensor())
        transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        to_input_transf = transforms.Compose(transf_list_batch)

        transf_list = []
        transf_list.append(transforms.Resize(256))
        transf_list.append(transforms.CenterCrop(224))
        transform = transforms.Compose(transf_list)

        encoder_image = EncoderCNN(512, 0.3, 'resnet101')
        # encoder_image = encoder_image.cuda()
        encoder_image = encoder_image

        path = os.path.join(image_dir, image_name)

        if os.path.exists(path):
            image = Image.open(path).convert('RGB')

            image_transf = transform(image)
            image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

            feature = encoder_image.forward(image_tensor).view(-1)
            features = [feature]
            feature = torch.stack(features)
            print('resnet:',time.time() - start)

            pred_fat = model_fat(feature)
            pred_salt = model_salt(feature)
            pred_sugars = model_sugars(feature)
            pred_saturates = model_saturates(feature)

            print('pred:',time.time() - start)

            _, pred_fat = torch.max(pred_fat, 1)
            _, pred_salt = torch.max(pred_salt, 1)
            _, pred_sugars = torch.max(pred_sugars, 1)
            _, pred_saturates = torch.max(pred_saturates, 1)
            dict_ = {"fat":pred_fat.item(), "salt":pred_salt.item(), "sugars":pred_sugars.item(), "saturates":pred_saturates.item()}
            # print(dict_)
            return dict_
        # return dict_
    # return Null
