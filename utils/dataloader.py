import random
import torch
import torch.utils.data as data
import os
import pickle as pk
import numpy as np
from ingrs_vocab import Vocabulary
import lmdb
from PIL import Image

class RecipeDataset(data.Dataset):
    def __init__(self,dir_file,transform,max_num_samples=-1,maxnumims=5,max_num_labels=20,max_unit_len=150,split='train'):
        
        self.maxnumims = maxnumims              # maximum number of images per recipe
        self.max_num_labels = max_num_labels    # max num of ingredients per recipe
        self.max_unit_len = max_unit_len
        self.split = split                      # train , val or test

        self.dir_file = dir_file
        self.transform = transform
               
        #self.ingrs_vocab = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_ingrs.pkl'), 'rb'))  # ingredients and their indexes
        self.vocab_unit = pk.load(open(os.path.join(dir_file,'recipe1m_vocab_unit.pkl'),'rb'))     # units , tokens and their indexes
        self.dataset = pk.load(open(os.path.join(dir_file,'filter_'+ split +'.pkl'), 'rb'))      # every recipe (id,instructions,tokenized,ingredients,images,title)

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

        # read lmdb file
        self.image_file = lmdb.open(os.path.join(self.dir_file, 'lmdb_' + split), max_readers=1, readonly=True,\
                                        lock=False, readahead=False, meminit=False)

        

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

        # random the idx array [0,1,2,3,4]
        idx_array = np.arange(len(img_paths))
        random.shuffle(idx_array)

        with self.image_file.begin(write=False) as txn:
            for i in idx_array:
                # the chosen img.jpg
                path = img_paths[i]
                image = txn.get(path.encode())
                if image :  # we got img that exist in lmdb                   
                    break
            image = np.fromstring(image, dtype=np.uint8)
            image = np.reshape(image, (256, 256, 3))
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transform is not None:
            image = self.transform(image)

        unit_idx = np.ones(self.max_unit_len) * self.vocab_unit('<pad>')

        pos2 = 0

        # add title to idx
        for title_word in title:
            idx = self.vocab_unit(title_word)
            unit_idx[pos2] = idx
            pos2 += 1

        # add unit to idx
        for sentence in ingrs_unit:
            if pos2 >= self.max_unit_len:
                break
            # add '<eoi>' between sentences
            unit_idx[pos2] = self.vocab_unit('<eoi>')
            pos2 += 1
            for token in sentence.split('_'):
                if pos2 >= self.max_unit_len:
                    break
                idx = self.vocab_unit(token)
                unit_idx[pos2] = idx
                pos2 += 1
                
        if pos2 < self.max_unit_len:
            unit_idx[pos2] = self.vocab_unit('<end>')


        labels_unit = torch.from_numpy(unit_idx).long()

        return image,labels_unit #,labels

    def __len__(self):
        return len(self.ids)

    def get_ingrs_vocab_size(self):
        return len(self.vocab_unit)

def get_loader(transform,dir_file='/home/r8v10/git/InvCo/dataset/',split='train',batch_size=4,shuffle=False,num_workers=1,drop_last=False):
    dataset = RecipeDataset(dir_file=dir_file,transform=transform,split=split,max_num_samples=-1)

    RecipeLoader = data.DataLoader(dataset=dataset,\
                                   batch_size=batch_size,\
                                   shuffle=shuffle,\
                                   num_workers=num_workers,\
                                   drop_last=drop_last)

    return RecipeLoader,dataset