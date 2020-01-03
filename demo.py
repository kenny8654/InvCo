import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from PIL import Image
import time
from utils.output_utils import prepare_output
from ingrs_vocab import Vocabulary

dir_file = '/home/r8v10/git/InvCo/dataset'
# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

ingrs_vocab = pickle.load(open(os.path.join(dir_file, 'recipe1m_vocab_unit.pkl'), 'rb'))
ingr_vocab_size = len(ingrs_vocab)

t = time.time()
import sys; sys.argv=['']; del sys
args = get_parser()
args.maxseqlen = 15
args.ingrs_only=False
model = get_model(args, ingr_vocab_size)

# Load the trained model parameters
model_dir = '/home/r8v10/git/InvCo/dataset/final_model/inversecooking/model/checkpoints'
model_path = os.path.join(model_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = False
model.recipe_only = False
print ('loaded model')
print ("Elapsed time:", time.time() -t)

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)

import requests
from io import BytesIO
import random
from collections import Counter
# set to true to load images from demo_urls instead of those in test_imgs folder
use_urls = True
#if True, it will show the recipe even if it's not valid
show_anyways = True
image_folder = '/home/r8v10/git/InvCo/dataset/Recipe1M/train/0/0/0/0'

if not use_urls:
    demo_imgs = os.listdir(image_folder)
    random.shuffle(demo_imgs)

demo_urls = ['http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/48/80/87/pictQbup8.jpg?fbclid=IwAR1UdhFEB3XfK3BFkS5q-Q6tZcVMGuT14kqL8zW6w324jCMbz-wuCfXIMZQ']

demo_files = demo_urls if use_urls else demo_imgs
print(demo_files)

for img_file in demo_files:
    if use_urls:
        response = requests.get(img_file)
        image = Image.open(BytesIO(response.content))
    else:
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path).convert('RGB')

    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    num_valid = 1
    #for i in range(numgens):
    i = 0
    while True:
        i += 1
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i],
                    temperature=temperature, beam=beam[i], true_ingrs=None)

        #ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
        # print(recipe_ids[0])
        outs, valid = prepare_output(recipe_ids[0], ingrs_vocab)

        if valid['is_valid'] or show_anyways:
            if valid['reason'] == 'All ok.':
                print ('RECIPE', num_valid)
                num_valid+=1

                BOLD = '\033[1m'
                END = '\033[0m'
                print (BOLD + '\nTitle:' + END,outs['title'])

                print (BOLD + '\nInstructions:'+END)
                print ('-'+'\n-'.join(outs['recipe']))
                print ('='*20)

                print ("Reason: ", valid['reason'])
                break
        #else:
        #    pass
        #    print ("Not a valid recipe!")
        #    print ("Reason: ", valid['reason'])



