import json
import os
import tqdm
import pickle

dir_ = '../dataset'
image_dir = '../dataset/Recipe1M'
file_ = 'nutritional_item.pkl'

pk = pickle.load(open(os.path.join(dir_, file_), 'rb'))

items = []

for i in tqdm.tqdm(pk):
    images = []
    item = {}
    for j in i['images']:
        path = os.path.join(image_dir, i['partition'], j[0], j[1], j[2], j[3], j)
        #print(path)
        #print(os.path.isfile(path))
        if os.path.isfile(path):
            images.append(path)
            print('true')
    item['lights'] = i['lights']
    item['images'] = images

pickle.dump(out, os.path.join(dir_,'nutritional_with_images.pkl'))
