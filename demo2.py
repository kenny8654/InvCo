import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
import json
from model import get_model
from torchvision import transforms
from PIL import Image
import time
from utils.output_utils import prepare_output
from ingrs_vocab import Vocabulary
import requests
from io import BytesIO
import random
from collections import Counter
from invco import ROOT_DIR

def main(dir_file, image_folder, demo_path, lights):
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
    model_dir = '/home/r8v10/git/InvCo/dataset/new_model/inversecooking/model/checkpoints'
#     model_dir = F'{ROOT_DIR}/dataset/model/inversecooking/model/checkpoints'
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

    # set to true to load images from demo_urls instead of those in test_imgs folder
    use_urls = False
    #if True, it will show the recipe even if it's not valid
    show_anyways = True

    if use_urls:
        response = requests.get(demo_path)
        image = Image.open(BytesIO(response.content))
    else:
        image_path = os.path.join(image_folder, demo_path)
        image = Image.open(image_path).convert('RGB')

    # print('Data path:', image_path)

    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    num_valid = 1
    temperature = 1.0
    # greedy = [True, False, False, False]
    # beam = [-1, -1, -1, -1]
    
    while True:
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=False,
                    temperature=temperature, beam= -1, true_ingrs=None)

        recipe_ids = outputs['recipe_ids'].cpu().numpy()
        
        outs, valid = prepare_output(recipe_ids[0], ingrs_vocab)
        num_valid+=1

        if valid['is_valid'] or show_anyways:
            if valid['reason'] == 'All ok.':
                # print ('RECIPE', num_valid)

                # BOLD = '\033[1m'
                # END = '\033[0m'
                # print (BOLD + '\nTitle:' + END,outs['title'])

                # print (BOLD + '\nInstructions:'+END)
                # print ('-'+'\n-'.join(outs['recipe']))
                # print ('='*20)

                #print ("Reason: ", valid['reason'])
                break
    recommend_id, recommend_lights = search(dir_file, outs['recipe'], lights)
    recommend_title, recommend_url = get_recipe(dir_file,recommend_id)

    # print('Recommendation of recipe:', recommend_id)
    # print('Title:', recommend_title)
    # print('Lights:', recommend_lights)
    # print('url:', recommend_url)




    return outs['title'], outs['recipe'], recommend_lights, recommend_title, recommend_url

def get_recipe(dir_file,recommend_id):
    with open(os.path.join(dir_file,'Recipe1M','recipes_with_nutritional_info.json'), 'r') as f:
        data = json.load(f)

    title = ''
    url = ''

    for i, item in enumerate(data):
        if item['id'] == recommend_id:
            found = True
            title = item['title']
            url = item['url']
            break
    
    return title, url



    
def search(dir_file, recipe, lights_txt):
    with open(os.path.join(dir_file,'nutritional_item.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    with open(os.path.join(dir_file,'ingr_vocab.pkl'), 'rb') as f:
        base = pickle.load(f)
    part = []

    for ingr in recipe:
        ingr_split = ingr.split(' ')
        ingr = ' '.join(text for text in ingr_split if text != 'teaspoon')

        for base_word in base:
            if base_word in ingr:
                part.append(base_word)
    lights_dic = eval(lights_txt)
    max_overlap = 0
    min_light_num = sum(lights_dic.values())
    min_light_txt = []
    recommend_list = 'None'
    recommend = ''
    lights = {'green':0, 'orange':1, 'red':2}

    for i, item in enumerate(data):
        overlap = 0
        for ingr in item['ingredients']:
            for base_word in part:
                if base_word in ingr:
                    overlap += 1
                    break

        if overlap > max_overlap:
            txt_light = [item['lights']['fat'], item['lights']['salt'], item['lights']['saturates'], item['lights']['sugars']]
            num_light = encode(txt_light, lights)

            if num_light < min_light_num:
                recommend_list = item['id']
                partition = item['partition']
                max_overlap = overlap
                min_light_num = num_light
                min_light_txt = txt_light

    return recommend_list, min_light_txt


def encode(txt,lights):
    return sum(lights[x] for i, x in enumerate(txt))

class Demo():
    def __init__(self):
        self.dir_file = F'{ROOT_DIR}/dataset'
        self.image_folder = F'{ROOT_DIR}/web/media/img'

    def demo(self, demo_path, lights):
        #dir_file = '/home/r8v10/git/InvCo/dataset'
        # Image directory path
        # image_folder = '/home/r8v10/git/InvCo/dataset/Recipe1M/train/0/0/0/0'
        
        # Input image name
        # demo_path = input("Input image path: ")
        # lights = input('Input light:')

        title, recipe, recommend_lights, recommend_title, recommend_url = main(self.dir_file, self.image_folder, demo_path, lights)
        output = {"title":title,"recipe":recipe,"recommend_lights":recommend_lights,"recommend_title":recommend_title,"recommend_url":recommend_url}
        # print(title, recipe, recommend_lights, recommend_title, recommend_url)
        return(output)
