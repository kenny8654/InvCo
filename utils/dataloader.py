import random
import torch
import torch.utils.data as data
import os
import pickle as pk
import numpy as np
from utils.ingrs_vocab import Vocabulary

class RecipeDataset(data.Dataset):
    def __init__(self,dir_file,max_num_samples=-1,maxnumims=5,max_num_labels=20,max_unit_len=150,split='train'):
        
        self.maxnumims = maxnumims              # maximum number of images per recipe
        self.max_num_labels = max_num_labels    # max num of ingredients per recipe
        self.max_unit_len = max_unit_len
        self.split = split                      # train , val or test

        self.dir_file = dir_file
               
        #self.ingrs_vocab = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_ingrs.pkl'), 'rb'))  # ingredients and their indexes
        self.vocab_unit = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_unit.pkl'),'rb'))     # units , tokens and their indexes
        self.dataset = pk.load(open(os.path.join(dir_file,'recipe1m_'+ split +'.pkl'), 'rb'))      # every recipe (id,instructions,tokenized,ingredients,images,title)

        # filter recipes that don't have image 
        self.ids = []
        for i, entry in enumerate(self.dataset): 
            if len(entry['images']) == 0:        # entry['images'] : ['XXXXX.jpg','XXXXX.jpg',...] 
                continue
            self.ids.append(i)                   # ids save all recipes with images
        
        #print('ids:',self.ids)
        print('len(ids):',len(self.ids))

        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples] 

    def __getitem__(self,index):

        # get a recipe according to index
        recipe = self.dataset[self.ids[index]]

        # read attributes from this recipe
        recipe_id = recipe['id']
        img_paths = recipe['images'][0:self.maxnumims]
        
        #ingrs = recipe['ingredients']
        title = recipe['title']

        # ingredients with units
        ingrs_unit = recipe['ingrs_unit'] 

        # shuffle the images
        if self.split == 'train':
            img_idx = np.random.randint(0, len(img_paths))
        else:
            img_idx = 0

        # the chosen img
        path = img_paths[img_idx]
        
        image_dir = os.path.join(self.dir_file,path[0], path[1], path[2], path[3], path)

        #print('image_dir:',image_dir)

        image = self.read_img(image_dir)

        unit_idx = np.ones(self.max_unit_len) * self.vocab_unit('<pad>')
        pos2 = 0
        # unit to idx
        for sentence in ingrs_unit:
            for token in sentence.split('_'):
                idx = self.vocab_unit(token)
                #print(idx,' : ',self.vocab_unit.idx2word[idx])
                unit_idx[pos2] = idx
                pos2 += 1
                if pos2 >= self.max_unit_len:
                    break

        labels_unit = torch.from_numpy(unit_idx).long()

        return image,labels_unit #,labels

    def __len__(self):
        return len(self.ids)

    def read_img(self,img_dir):
        #print('read img from:',img_dir)
        # add method to read img here :



        img = img_dir
        return img

def get_loader(dir_file,split,batch_size=4,shuffle=False):
    dataset = RecipeDataset(dir_file,split=split)

    RecipeLoader = data.DataLoader(dataset=dataset,\
                                   batch_size=batch_size,\
                                   shuffle=shuffle)

    return RecipeLoader,dataset
