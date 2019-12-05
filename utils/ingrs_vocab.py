import nltk
import pickle
import argparse
from collections import Counter
import json
import os
import numpy as np
import re
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

        for i, entry in enumerate(layer1):
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

def cluster_vocab(thesaurus):
    vocab = dict()
    ingr_category = dict()

def main(dir_file):
    build_vocab(dir_file)

if __name__ == '__main__':
    dir_file = F'{DATASET_DIR}/recipe1M_layers'
    main(dir_file)