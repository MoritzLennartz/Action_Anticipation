#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lachaji
"""

import numpy as np
import time

from utils.pie_data import PIE
from utils.preprocessing import *

import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.ted_train_utils import train, test
from models.torch_transformer_1d_full import Transformer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import json


seed = 90
seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIE_PATH = 'PIE_dataset'

data_opts = {'fstride': 1,
        'sample_type': 'all', 
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'random',  #  kfold, random, default
        'seq_type': 'crossing', #  crossing , intention
        'min_track_size': 15, #  discard tracks that are shorter
        'kfold_params': {'num_folds': 1, 'fold': 1},
        'random_params': {'ratios': [0.7, 0.15, 0.15],
                                    'val_data': True,
                                    'regen_data': False},
        'tte' : [30, 60],
        'batch_size': 16
        }
        


imdb = PIE(data_path=PIE_PATH)
seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
balanced_seq_train = balance_dataset(seq_train)
obs_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, data_opts['tte'], 0.8, 16)


seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
balanced_seq_valid = balance_dataset(seq_valid)
obs_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid,  data_opts['tte'], 0, 16)


seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
obs_seq_test, traj_seq_test = tte_dataset(seq_test, data_opts['tte'], 0, 16)


bbox_enc_train = obs_seq_train['bbox']
bbox_enc_valid = obs_seq_valid['bbox']
bbox_enc_test  = obs_seq_test['bbox']


bbox_dec_train = traj_seq_train['bbox']
bbox_dec_valid = traj_seq_valid['bbox']
bbox_dec_test  = traj_seq_test['bbox']

action_train = obs_seq_train['activities']
action_valid = obs_seq_valid['activities']
action_test  = obs_seq_test['activities']

normalized_bbox_enc_train = normalize_bbox(bbox_enc_train)
normalized_bbox_enc_valid = normalize_bbox(bbox_enc_valid)
normalized_bbox_enc_test  = normalize_bbox(bbox_enc_test)


normalized_bbox_dec_train = normalize_bbox(bbox_dec_train)
normalized_bbox_dec_valid = normalize_bbox(bbox_dec_valid)
normalized_bbox_dec_test  = normalize_bbox(bbox_dec_test)

label_action_train = prepare_label(action_train)
label_action_valid = prepare_label(action_valid)
label_action_test = prepare_label(action_test)


X_train_enc, X_valid_enc = torch.Tensor(normalized_bbox_enc_train), torch.Tensor(normalized_bbox_enc_valid)
Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
X_test_enc, Y_test = torch.Tensor(normalized_bbox_enc_test), torch.Tensor(label_action_test)


X_train_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_train, 60))
X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60))
X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

trainset = TensorDataset(X_train_enc, X_train_dec, Y_train)
validset = TensorDataset(X_valid_enc, X_valid_dec, Y_valid)
testset = TensorDataset(X_test_enc, X_test_dec, Y_test)


train_loader = DataLoader(trainset, batch_size = input_opts['batch_size'], shuffle = True)
valid_loader = DataLoader(validset, batch_size = input_opts['batch_size'], shuffle = True)
test_loader = DataLoader(testset, batch_size = 256)


print("Start Testing \n")

model_folder_name = 'TED'
f = open("checkpoints/{}.json".format(model_folder_name),"r")
input_opts = json.loads(f.read())


reg_lambda = input_opts['reg_lambda']
cl_lambda = input_opts['cl_lambda']

checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
writer = SummaryWriter('logs/{}'.format(model_folder_name))


model = Transformer(input_opts['num_layers'], input_opts['d_model'],
                                input_opts['d_input'], input_opts['num_heads'], input_opts['dff'],
                                input_opts['pos_encoding_enc'], input_opts['pos_encoding_dec'])


model.to(device)

checkpoint = torch.load(checkpoint_filepath, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


pred, lab = test(model, test_loader)
pred_cpu = torch.Tensor.cpu(pred)
lab_cpu = torch.Tensor.cpu(lab)
acc = accuracy_score(lab_cpu, np.round(pred_cpu))
conf_matrix = confusion_matrix(lab_cpu, np.round(pred_cpu), normalize = 'true')
f1 = f1_score(lab_cpu, np.round(pred_cpu))
auc = roc_auc_score(lab_cpu, np.round(pred_cpu))

input_opts['acc'] = acc
input_opts['f1'] = f1
input_opts['conf_matrix'] = str(conf_matrix)
input_opts['auc'] = auc
config = json.dumps(input_opts)


f = open("checkpoints/{}.json".format(model_folder_name),"w")
f.write(config)
f.close()

print(f"Accuracy: {acc} \n AUC: {auc} \n f1: {f1} ")
