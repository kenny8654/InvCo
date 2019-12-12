import nltk
import pickle
import argparse
from collections import Counter
import json
import os
import numpy as np
import re
from tqdm import tqdm
nltk.download('punkt')
# Local import
from invco import DATASET_DIR


def get_ingredient(det_ingr, replace_dict):
    det_ingr_lower = det_ingr['text'].lower()
    
    det_ingr_lower = ''.join(i for i in det_ingr_lower if not i.isdigit())
    
    #########################Layer1 ingredients##########################
    #remove info in brackets
    # regex = '\(.*?\)'
    # det_ingr_lower = re.sub(regex,'',det_ingr_lower)
    #####################################################################

    #replace word
    for rep_word, char_list in replace_dict.items():
        for char in char_list:
            if char in det_ingr_lower:
                det_ingr_lower = det_ingr_lower.replace(char, rep_word)
    #remove trailing newline
    det_ingr_lower = det_ingr_lower.strip()
    det_ingr_lower = det_ingr_lower.replace(' ', '_')

    return det_ingr_lower

def get_instruction(instr, replace_dict_instrs):
    instr_lower = instr['text'].lower()

    #replace word
    for rep_word, char_list in replace_dict_instrs.items():
        for char in char_list:
            if char in instr_lower:
                instr_lower = instr_lower.replace(char, rep_word)
        instr_lower = instr_lower.strip()
    #remove sentences starting with "1.", "2.", ... from the targets
    if len(instr_lower) > 0 and instr_lower[0].isdigit():
        instr_lower = ''
    return instr_lower

def update_counter(list_, counter_toks, istrain=False):
    for sentence in list_:
        tokens = nltk.tokenize.word_tokenize(sentence)
        if istrain:
            counter_toks.update(tokens)

def cluster_vocab(counter_ingrs):
    ingr_mydict = dict()
    ingr_category = dict()
    
    for key,value in counter_ingrs.items():
        if len(key.split('_')) > 1:
            w1 = key.split('_')[-1]
            w2 = key.split('_')[0]
            w3 = key.split('_')[-2] + '_' + key.split('_')[-1]
            w4 = key.split('_')[0] + '_' + key.split('_')[1]
            word_candidate = [w1, w2, w3, w4]
        else:
            w1 = key.split('_')[-1]
            w2 = key.split('_')[0]
            word_candidate = [w1, w2]


        #find the smaller ingredient category
        found = False
        for word in word_candidate:
            if word in counter_ingrs.keys():
                part = word.split('_')
                if len(part) > 1:
                    if part[0] in counter_ingrs.keys():
                        word = part[0]
                    elif part[1] in counter_ingrs.keys():
                        word = part[1]
                if word in ingr_mydict.keys():
                    ingr_mydict[word] += value
                    ingr_category[word].append(key)
                else:
                    ingr_mydict[word] = value
                    ingr_category[word] = [key]
                
                found = True
                break

        if not found:
            ingr_mydict[key] = value
            ingr_category[key] = [key]
        
    print('Size of pepper ingredient:', len(ingr_category['pepper']))
    print('Size of cheese ingredient:', len(ingr_category['cheese']))

    return ingr_mydict,ingr_category

def remove_plurals(counter_ingrs, category_ingrs):
    del_ingrs = []

    for key, value in counter_ingrs.items():

        if len(key) == 0:
            del_ingrs.append(key)
            continue

        found = False
        if key[-2:] == 'es':
            if key[:-2] in counter_ingrs.keys():
                counter_ingrs[key[:-2]] += value
                category_ingrs[key[:-2]].extend(category_ingrs[key])
                del_ingrs.append(key)
                found = True

        elif key[-1] == 's' and not found:
            if key[:-1] in counter_ingrs.keys():
                counter_ingrs[key[:-1]] += value
                category_ingrs[key[:-1]].extend(category_ingrs[key])
                del_ingrs.append(key)

    for item in del_ingrs:
        del counter_ingrs[item]
        del category_ingrs[item]
    return counter_ingrs, category_ingrs

def build_vocab(dir_file):
    
    #open file
    print('Loading data')
    dets = json.load(open(os.path.join(dir_file, 'det_ingrs.json'), 'r'))
    layer1 = json.load(open(os.path.join(dir_file, 'layer1.json'), 'r'))
    layer2 = json.load(open(os.path.join(dir_file, 'layer2.json'), 'r'))

    #save rec_id via line
    id2im_pos = {}
    for i, line in enumerate(layer2):
        id2im_pos[line['id']] = i
        
    id2ingr_pos = {}
    for i, line in enumerate(dets):
        id2ingr_pos[line['id']] = i
    

    print('Data loaded!')

    #replace instruction and ingredients words 
    replace_Ingrs = {'and': ['&',"'n"], '':  ['%', ',', '.', '#', '[', ']', '!', '?']}
    replace_dict_instrs = {'and': ['&', "'n"], '': ['#', '[', ']']}
    
    #Count words
    ingrs_file = os.path.join(dir_file, 'allingrs_count.pkl')
    instrs_file = os.path.join(dir_file, 'allinstrs_count.pkl')

    if os.path.exists(ingrs_file) and os.path.exists(instrs_file):
        print('File existed!')
        with open('allingrs_count.pkl', 'rb') as file:
            counter_ingrs = pickle.load(file)
        with open('allinstrs_count.pkl', 'rb') as file:
            counter_toks = pickle.load(file)
    else:
        counter_toks = Counter()
        counter_ingrs = Counter()

        for i, entry in tqdm(enumerate(layer1)):
            #get all instructions from the recipe
            instrs = entry['instructions']

            instrs_list = []
            ingrs_lsit = []

            #get ingredients
            det_ingrs = dets[id2ingr_pos[entry['id']]]['ingredients']
            valid = dets[id2ingr_pos[entry['id']]]['valid']

            for j,det_ingr in enumerate(det_ingrs):
                if len(det_ingr) > 0 and valid[j]:
                    ingrs_lsit.append(get_ingredient(det_ingr, replace_Ingrs))
            
            #get instructions
            acc_len = 0 #Calculate length of instruction
            for j,instr in enumerate(instrs):
                tmp = get_instruction(instr, replace_dict_instrs)
                if len(tmp) > 0:
                    instrs_list.append(tmp)
                    # acc_len += len(tmp)
            
            #remove recipes with too few or too many ingredients or instruction
            if len(ingrs_lsit) < 2 or len(instrs_list) < 2 \
                    or len(ingrs_lsit) > 20 or len(instrs_list) >20:
                continue

            #tokenize sentences and update counter
            update_counter(instrs_list, counter_toks, istrain=entry['partition'] == 'train')
            if entry['partition'] == 'train':
                counter_ingrs.update(ingrs_lsit)
        
        pickle.dump(counter_ingrs, open(ingrs_file, 'wb'))
        pickle.dump(counter_toks, open(instrs_file, 'wb'))
    
    #Pre-custom thesaurus
    with open ('ingr_vocab.pkl', 'rb') as file:
        base_words = pickle.load(file)

    counter_ingrs.update(base_words)

    #simplified ingredient
    print('original data %d' %(len(counter_ingrs)))
    counter_ingrs, category_ingrs = cluster_vocab(counter_ingrs)
    counter_ingrs, category_ingrs = remove_plurals(counter_ingrs, category_ingrs)

    print('simplified ingredient %d' %(len(counter_ingrs)))

    # If the word frequency is less than '10', then the word is discarded.
    words = [word for word, count in counter_toks.items() if count >= 10]
    ingrs = {word: count for word, count in counter_ingrs.items() if count >= 10}

    print('Length of ingredients %d' % (len(ingrs)))


    # Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_toks = Vocabulary()
    vocab_toks.add_word('<start>')
    vocab_toks.add_word('<end>')
    vocab_toks.add_word('<eoi>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab_toks.add_word(word)
    vocab_toks.add_word('<pad>')

    # Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    idx = vocab_ingrs.add_word('<end>')
    # this returns the next idx to add words to
    # Add the ingredients to the vocabulary.
    for k, _ in ingrs.items():
        for ingr in category_ingrs[k]:
            idx = vocab_ingrs.add_word(ingr, idx)
        idx += 1
    _ = vocab_ingrs.add_word('<pad>', idx)

    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    print("Total token vocabulary size: {}".format(len(vocab_toks)))

    dataset = {'train': [], 'val': [], 'test': []}

    for i, entry in tqdm(enumerate(layer1)):

        # get all instructions for this recipe
        instrs = entry['instructions']

        instrs_list = []
        ingrs_list = []
        images_list = []

        # retrieve pre-detected ingredients for this entry
        det_ingrs = dets[id2ingr_pos[entry['id']]]['ingredients']
        valid = dets[id2ingr_pos[entry['id']]]['valid']
        labels = []

        for j, det_ingr in enumerate(det_ingrs):
            if len(det_ingr) > 0 and valid[j]:
                det_ingr_undrs = get_ingredient(det_ingr, replace_Ingrs)
                ingrs_list.append(det_ingr_undrs)
                label_idx = vocab_ingrs(det_ingr_undrs)
                if label_idx is not vocab_ingrs('<pad>') and label_idx not in labels:
                    labels.append(label_idx)

        # get raw text for instructions of this entry
        acc_len = 0
        for instr in instrs:
            instr = get_instruction(instr, replace_dict_instrs)
            if len(instr) > 0:
                acc_len += len(instr)
                instrs_list.append(instr)

        # we discard recipes with too many or too few ingredients or instruction words

        if len(labels) < 2 or len(instrs_list) < 2 \
                or len(instrs_list) >= 20 or len(labels) >= 20 \
                or acc_len < 20:
            continue

        if entry['id'] in id2im_pos.keys():
            ims = layer2[id2im_pos[entry['id']]]

            # copy image paths for this recipe
            for im in ims['images']:
                images_list.append(im['id'])

        # tokenize sentences
        toks = []

        for instr in instrs_list:
            tokens = nltk.tokenize.word_tokenize(instr)
            toks.append(tokens)

        title = nltk.tokenize.word_tokenize(entry['title'].lower())

        newentry = {'id': entry['id'], 'instructions': instrs_list, 'tokenized': toks,
                    'ingredients': ingrs_list, 'images': images_list, 'title': title}
        dataset[entry['partition']].append(newentry)

    print('Dataset size:')
    for split in dataset.keys():
        print(split, ':', len(dataset[split]))

    return vocab_ingrs, vocab_toks, dataset

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word, idx=None):
        if idx is None:
            if not word in self.word2idx.keys():
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if not word in self.word2idx.keys():
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]
                return idx

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def main(dir_file):
    # build_vocab(dir_file)
    vocab_ingrs, vocab_toks, dataset = build_vocab(dir_file)

    with open(os.path.join(dir_file + 'recipe1m_vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)
    with open(os.path.join(dir_file +'recipe1m_vocab_toks.pkl'), 'wb') as f:
        pickle.dump(vocab_toks, f)

    for split in dataset.keys():
        with open(os.path.join(dir_file + 'recipe1m_' + split + '.pkl'), 'wb') as f:
            pickle.dump(dataset[split], f)

if __name__ == '__main__':
    dir_file = F'{DATASET_DIR}/recipe1M_layers'
    main(dir_file)