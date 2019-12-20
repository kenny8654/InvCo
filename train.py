from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import random
import pickle
from utils.dataloader import get_loader
from ingrs_vocab import Vocabulary
from model import get_model
from torchvision import transforms
import sys
import json
import time
import torch.backends.cudnn as cudnn
# from utils.tb_visualizer import Visualizer
# from model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main(args):
    
    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(args.save_dir, args.project_name, 'tb_logs', args.model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)

    # if args.tensorboard:
    #     logger = Visualizer(tb_logs, name='visual_results')

    # check if we want to resume from last checkpoint of current model
    if args.resume:
        args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
        args.resume = True

    # logs to disk
    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')

    print(args)
    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

    # patience init
    curr_pat = 0

    # Build data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.recipe1m_dir

    for split in ['train', 'val']:

        transforms_list = [transforms.Resize((args.image_size))]

        if split == 'train':
            # Image preprocessing, normalization for the pretrained resnet
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(transforms.RandomCrop(args.crop_size))

        else:
            transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))

        transform = transforms.Compose(transforms_list)
        max_num_samples = max(args.max_eval, args.batch_size) if split == 'val' else -1
        data_loaders[split], datasets[split] = get_loader(transform, data_dir, 
                                                          split, args.batch_size,
                                                          shuffle=split == 'train', num_workers=args.num_workers,
                                                          drop_last=True,)
        
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    print('Length of ingredients:', ingr_vocab_size)

    # Build the model
    model = get_model(args, ingr_vocab_size)
    keep_cnn_gradients = False

    decay_factor = 1.0

    # add model parameters
    params = list(model.recipe_decoder.parameters()) 

    # only train the linear layer in the encoder if we are not transfering from another model
    if args.transfer_from == '':
        params += list(model.image_encoder.linear.parameters())
    params_cnn = list(model.image_encoder.resnet.parameters())

    print ("CNN params:", sum(p.numel() for p in params_cnn if p.requires_grad))
    print ("decoder params:", sum(p.numel() for p in params if p.requires_grad))

    # start optimizing cnn from the beginning
    if params_cnn is not None and args.finetune_after == 0:
        optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn,
                                                           'lr': args.learning_rate*args.scale_learning_rate_cnn}],
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
        keep_cnn_gradients = True
        print ("Fine tuning resnet")
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.resume:
        model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'model.ckpt')
        optim_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'optim.ckpt')
        optimizer.load_state_dict(torch.load(optim_path, map_location=map_loc))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.load_state_dict(torch.load(model_path, map_location=map_loc))
        

if __name__ == '__main__':
    args = get_parser()
    main(args)