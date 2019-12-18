import nltk
import pickle
import argparse
from collections import Counter
from itertools import zip_longest
import json
import os
from tqdm import tqdm
import numpy as np
import re
nltk.download('punkt')
# Local import
from invco import DATASET_DIR

def get_ingredient(det_ingr, replace_dict):
    det_ingr_lower = det_ingr['text'].lower()
    
    det_ingr_lower = ''.join(i for i in det_ingr_lower if not i.isdigit())

    #replace word
    for rep_word, char_list in replace_dict.items():
        for char in char_list:
            if char in det_ingr_lower:
                det_ingr_lower = det_ingr_lower.replace(char, rep_word)
    
    #remove trailing newline
    det_ingr_lower = det_ingr_lower.strip()
    det_ingr_lower = det_ingr_lower.replace(' ', '_')

    return det_ingr_lower

def get_units(det_unit, replace_word):
    det_unit_lower = det_unit['text'].lower()
    det_unit_lower = ''.join(i for i in det_unit_lower)

    #remove info in brackets
    regex = '\(.*?\)'
    det_unit_lower = re.sub(regex,'',det_unit_lower)

    # remove the redundant quantity
    found_digit = False
    ingrs = ''
    for item in det_unit_lower.split(' '):
        if not len(item) > 0:
            continue
        if not item[0].isdigit() or not found_digit:
            found_digit = True
            ingrs += item+' '
    
    #replace word
    for rep_word, char_list in replace_word.items():
        for char in char_list:
            if char in ingrs:
                ingrs = ingrs.replace(char, rep_word)
    
    #remove trailing newline
    ingrs = ingrs.strip()
    ingrs = ingrs.replace(' ', '_')

    return ingrs

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

def update_counter(list_, counter_units, istrain=False):

    for sentence in list_:
        sentence = sentence.replace('_', ' ')
        tokens = nltk.tokenize.word_tokenize(sentence)
        if istrain:
            counter_units.update(tokens)

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

def plurals(ingr_candidate):
    if ingr_candidate[-2:] == 'es':
        ingr_candidate = ingr_candidate[:-2]

    elif ingr_candidate[-1:] == 's':
        ingr_candidate = ingr_candidate[:-1]
    
    return ingr_candidate

def build_vocab(dir_file):
    
    #open file
    print('Loading data')
    dets = json.load(open(os.path.join(dir_file, 'det_ingrs.json'), 'r'))
    layer1 = json.load(open(os.path.join(dir_file, 'layer1.json'), 'r'))
    layer2 = json.load(open(os.path.join(dir_file, 'layer2+.json'), 'r'))

    #save rec_id via line
    id2im_pos = {}
    for i, line in enumerate(layer2):
        id2im_pos[line['id']] = i
        
    id2ingr_pos = {}
    for i, line in enumerate(layer1):
        id2ingr_pos[line['id']] = i
    

    print('Data loaded!')

    #replace instruction and ingredients words 
    replace_Ingrs = {'and': ['&',"'n"], 'cup': ['c.', 'cups'], 'tablespoon': ['tbsp', 'tablespoons'], 'teaspoon': ['teaspoons', 'ts'], '':  ['%', ',', '.', '#', '[', ']', '!', '?']}
    replace_dict_instrs = {'and': ['&', "'n"], '': ['#', '[', ']']}
    # replace_units = {'and': ['&',"'n"], 'cup': ['c.', 'cups'], 'tablespoon': ['tbsp', 'tablespoons'], 'teaspoon': ['teaspoons', 'ts'], '':  ['%', ',', '.', '#', '[', ']', '!', '?', '-', 'to']}

    #Count words
    ingrs_file = os.path.join(dir_file, 'allingrs_count.pkl')
    unit_file = os.path.join(dir_file, 'allingrs_unit.pkl')
    
    if os.path.exists(ingrs_file) and os.path.exists(unit_file):    
        print('File existed!')
        with open('allingrs_count.pkl', 'rb') as file:
            counter_ingrs = pickle.load(file)
        with open('allingrs_unit.pkl', 'rb') as file:
            counter_units = pickle.load(file)
    else:
        dataset = {'train': [], 'val': [], 'test': []}
        counter_ingrs = Counter()
        counter_units = Counter()

        for i, entry in tqdm(enumerate(layer1)):
            #get all instructions from the recipe
            instrs = entry['instructions']

            instrs_list = []
            ingrs_list = []
            unit_list = []
            images_list = []

            #get ingredients
            det_ingrs = dets[id2ingr_pos[entry['id']]]['ingredients']
            valid = dets[id2ingr_pos[entry['id']]]['valid']
            det_units = layer1[id2ingr_pos[entry['id']]]['ingredients']
            ingrs_data = zip_longest(det_ingrs,det_units)

            for j, (det_ingr,det_unit) in enumerate(ingrs_data):
                if len(det_ingr) > 0 and valid[j]:
                    ingr_candidate = get_ingredient(det_ingr, replace_Ingrs)
                    ingr_candidate = plurals(ingr_candidate)
                    ingrs_list.append(ingr_candidate)
                    unit_lower = get_units(det_unit, replace_Ingrs)
                    if ingr_candidate in unit_lower:
                        unit_candidate = simplify_ingrs(unit_lower, ingr_candidate)
                        unit_list.append(unit_candidate)
                    else:
                        unit_candidate = '1_' + ingr_candidate
                        unit_list.append(unit_candidate)

            #get instruction
            acc_len = 0 #Calculate length of instruction
            for j,instr in enumerate(instrs):
                tmp = get_instruction(instr, replace_dict_instrs)
                if len(tmp) > 0:
                    instrs_list.append(tmp)
                    acc_len += len(tmp)

            
            # we discard recipes with too many or too few ingredients or instruction words
            if len(ingrs_list) < 2 or len(instrs_list) < 2 \
                    or len(instrs_list) >= 20 or len(ingrs_list) >= 20 \
                    or acc_len < 20:
                continue

            if entry['id'] in id2im_pos.keys():
                ims = layer2[id2im_pos[entry['id']]]

                # copy image paths for this recipe
                for im in ims['images']:
                    images_list.append(im['id'])

            #tokenize sentences and update counter
            # update_counter(instrs_list, counter_toks, istrain=entry['partition'] == 'train')
            update_counter(unit_list, counter_units, istrain=entry['partition'] == 'train')
            title = nltk.tokenize.word_tokenize(entry['title'].lower())

            if entry['partition'] == 'train':
                counter_ingrs.update(ingrs_list)
            if entry['partition'] == 'train':
                counter_units.update(title)
            

            newentry = {'id': entry['id'], 'ingredients': ingrs_list,
                        'ingrs_unit': unit_list,
                        'images': images_list, 'title': title}
            dataset[entry['partition']].append(newentry)
        
        pickle.dump(counter_ingrs, open(ingrs_file, 'wb'))
        pickle.dump(counter_units, open(unit_file, 'wb'))
    
    print('Dataset size:')
    for split in dataset.keys():
        print(split, ':', len(dataset[split]))    

    
    
    # Pre-custom thesaurus
    with open ('ingr_vocab.pkl', 'rb') as file:
        base_words = pickle.load(file)

    counter_ingrs.update(base_words)
    counter_units.update(base_words)

    #simplified ingredient
    print('original data %d' %(len(counter_ingrs)))
    counter_ingrs, category_ingrs = cluster_vocab(counter_ingrs)
    counter_ingrs, category_ingrs = remove_plurals(counter_ingrs, category_ingrs)

    print('simplified ingredient %d' %(len(counter_ingrs)))

    # If the word frequency is less than '10', then the word is discarded.
    ingrs = {word: count for word, count in counter_ingrs.items() if count >= 10}
    unit = [word for word, cnt in counter_units.items() if cnt >= 10]

    print('Length of ingredients %d' % (len(ingrs)))
    print('Length of unit ingredients %d' % (len(unit)))

    # Recipe unit vocab
    # Create a vocab wrapper and add some special tokens.
    unit_toks = Vocabulary()
    unit_toks.add_word('<start>')
    unit_toks.add_word('<end>')
    unit_toks.add_word('<eoi>')

    # Add the words to the vocabulary.
    for i, word in enumerate(unit):
        unit_toks.add_word(word)    
    unit_toks.add_word('<pad>')

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

    return vocab_ingrs, unit_toks, dataset

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

def simplify_ingrs(_unit, ingr_candidate):
    replace_units = {'cup', 'tablespoon', 'teaspoon', 'teaspoonp', 'lb', 'ounces', 'can', 'large', 'medium', 'whole',
                    'pound', 'pounds', 'small', 'package', 'g', 'lbs', 'ounce', 'pinch', 'grams', 'cans', 'can',
                    'ml', 'dash', 'pkg', 'oz', 'clove', 'bunch', 'packages', 'jar', 'box', 'quart', 'stick',
                    'bag', 'qt', 'tbs', 'bottle'}
    det_unit = _unit.lower()
    part = det_unit.split('_')
    unit_word = ''
    quantity = ''

    #find unit of ingredient
    for word in replace_units:
        if word in part:
            unit_word = word

    #find quantity of ingredient
    for i, item in enumerate(det_unit.split('_')):
        if len(item) > 0:
            if item[0].isdigit():
                quantity = part[i]
                break
    
    if len(quantity) > 0 and len(unit_word) > 0:
        det_unit = quantity + '_' + unit_word + '_' + ingr_candidate
    elif not len(quantity) > 0 and len(unit_word) > 0:
        det_unit = '1' + '_' + unit_word + '_' + ingr_candidate
    elif len(quantity) > 0 and not len(unit_word) > 0:
        det_unit = quantity + '_' + ingr_candidate
    else:
        det_unit = '1_' + ingr_candidate
    
    return det_unit

def main(dir_file):
    # build_vocab(dir_file)
    vocab_ingrs, vocab_unit, dataset = build_vocab(dir_file)

    with open(os.path.join(dir_file + 'recipe1m_vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)

    with open(os.path.join(dir_file + 'recipe1m_vocab_unit.pkl'), 'wb') as f:
        pickle.dump(vocab_unit, f)

    for split in dataset.keys():
        with open(os.path.join(dir_file + 'recipe1m_' + split + '.pkl'), 'wb') as f:
            pickle.dump(dataset[split], f)

if __name__ == '__main__':
    dir_file = F'{DATASET_DIR}/recipe1M_layers'
    main(dir_file)