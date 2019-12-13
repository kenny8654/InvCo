import random
import torch
import torch.utils.data as data
import os
import pickle as pk
import numpy as np
from utils.ingrs_vocab import Vocabulary

class RecipeDataset(data.Dataset):
    def __init__(self,dir_file,max_num_samples=-1,maxnumims=5,max_num_labels=20,split='train'):
        
        self.maxnumims = maxnumims              # maximum number of images per recipe
        self.max_num_labels = max_num_labels    # max num of ingredients per recipe
        self.split = split                      # train , val or test
               
        self.ingrs_vocab = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_ingrs.pkl'), 'rb'))  # ingredients and their indexes
        self.instrs_vocab = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_toks.pkl'), 'rb'))  # single words and their indexes
        self.dataset = pk.load(open(os.path.join(dir_file,'recipe1m_test.pkl'), 'rb'))             # every recipe (id,instructions,tokenized,ingredients,images,title)

        # filter recipes that don't have image 
        self.ids = []
        for i, entry in enumerate(self.dataset): 
            if len(entry['images']) == 0:        # entry['images'] : ['XXXXX.jpg','XXXXX.jpg',...] 
                continue
            self.ids.append(i)                   # ids save all recipes with images
        
        #print('ids:',self.ids)
        #print('len(ids):',len(self.ids))

        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples] 

    def __getitem__(self,index):

        # get a recipe according to index 
        recipe = self.dataset[self.ids[index]]

        # read attributes from this recipe
        recipe_id = recipe['id']
        instr_words = recipe['tokenized']
        img_paths = recipe['images'][0:self.maxnumims]
        ingrs = self.dataset[self.ids[index]]['ingredients']
        title = recipe['title']

        #print('recipe_id:',recipe_id)
        #print('instr_words:',instr_words)
        #print('img_paths:',img_paths)
        #print('ingrs:',ingrs)
        #print('title:',title)

        # init the index array
        ingrs_idx = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')

        pos = 0
        for i in range(self.max_num_labels):
            
            if i >= len(ingrs):                 # len(labels) : num of ingredients of this recipe
                ingr = '<pad>'  
            else :
                ingr = ingrs[i]
            label_idx = self.ingrs_vocab(ingr)  # get the index of this ingredient 
            
            # append(?) the index to array , and avoid duplication
            if label_idx not in ingrs_idx:
                ingrs_idx[pos] = label_idx
                pos += 1

        # add index of '<end>' (0) at the end of ingredients
        ingrs_idx[pos] = self.ingrs_vocab('<end>')

        #print('ingrs_idx:',ingrs_idx)

        labels = torch.from_numpy(ingrs_idx).long()

        #print('labels:',labels)

        if self.split == 'train':
            img_idx = np.random.randint(0, len(img_paths))
        else:
            img_idx = 0
        path = img_paths[img_idx]
        
        image_dir = os.path.join(path[0], path[1], path[2], path[3], path)

        #print('image_dir:',image_dir)

        image = self.read_img(image_dir)

        return image,labels

    def __len__(self):
        return len(self.ids)

    def read_img(self,img_dir):
        #print('read img from:',img_dir)
        # add method to read img here :



        img = img_dir
        return img

def get_loader(dir_file,batch_size=16,shuffle=False):
    dataset = RecipeDataset(dir_file,split='train')

    RecipeLoader = data.DataLoader(dataset=dataset,\
                                   batch_size=batch_size,\
                                   shuffle=shuffle)

    return RecipeLoader,dataset
