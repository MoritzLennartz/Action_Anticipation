#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:01:43 2021

@author: lachaji
"""

import numpy as np
import sys
import torch
import random

def balance_dataset(dataset, flip = True):
    
    print('\n#####################################')
    print('Generating balanced raw data')
    print('#####################################')
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'image_dimension': (1920, 1080)}  
    
    gt_labels = [gt[0] for gt in d['activities']]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples

    # finds the indices of the samples with larger quantity
    if num_neg_samples == num_pos_samples:
        print('Positive and negative samples are already balanced')  
        
    else:
        print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1
        else:
            gt_augment = 0
            
        
        img_width = d['image_dimension'][0]
        num_samples = len(d['pid'])
        
        for i in range(num_samples):
            if d['activities'][i][0][0] == gt_augment:
                flipped = d['center'][i].copy()
                flipped = [[img_width - c[0], c[1]]
                               for c in flipped]
                d['center'].append(flipped)
                
                flipped = d['bbox'][i].copy()
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]])
                               for c in flipped]
                d['bbox'].append(flipped)
                
                d['pid'].append(dataset['pid'][i].copy())
                
                d['activities'].append(d['activities'][i].copy())
                
                flipped = d['image'][i].copy()
                flipped = [c.replace('.png', '_flip.png') for c in flipped]
                
                d['image'].append(flipped)
                
        gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples
        
        if num_neg_samples > num_pos_samples:
            rm_index = np.where(np.array(gt_labels) == 0)[0]
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0]
            
        # Calculate the difference of sample counts
        dif_samples = abs(num_neg_samples - num_pos_samples)
        
        # shuffle the indices
        np.random.seed(42)
        np.random.shuffle(rm_index)
        # reduce the number of indices to the difference
        rm_index = rm_index[0:dif_samples]
        
        # update the data
        for k in d:
            seq_data_k = d[k]
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]    
    
        new_gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
        print('Balanced:\t Positive: %d  \t Negative: %d\n'
              % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n'
              % (len(d['activities'])))
        
    return d

def tte_dataset(dataset, time_to_event, overlap, obs_length):
    
    d_obs = {'bbox': dataset['bbox'].copy(),
     'pid': dataset['pid'].copy(),
     'activities': dataset['activities'].copy(),
     'image': dataset['image'].copy(),
     'center': dataset['center'].copy()
     } 

    d_tte = {'bbox': dataset['bbox'].copy(),
     'pid': dataset['pid'].copy(),
     'activities': dataset['activities'].copy(),
     'image': dataset['image'].copy(),
     'center': dataset['center'].copy()}     
    
    if isinstance(time_to_event, int):
        for k in d_obs.keys():   
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event]
                d_tte[k][i] = d_tte[k][i][- time_to_event:]
        d_obs['tte'] = [[time_to_event]]*len(dataset['bbox'])
        d_tte['tte'] = [[time_to_event]]*len(dataset['bbox'])
            
    else:
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res
        
        for k in d_obs.keys():
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)])
                seqs_tte.extend([seq[i + obs_length :] for i in range(start_idx, end_idx, olap_res)])
                d_obs[k] = seqs
                d_tte[k] = seqs_tte
        
        tte_seq = []
        for seq in dataset['bbox']:
            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            tte_seq.extend([[len(seq) - (i + obs_length)] for i in range(start_idx, end_idx , olap_res)])
            d_obs['tte'] = tte_seq.copy()
            d_tte['tte'] = tte_seq.copy()
            

    remove_index = []
    try:
        time_to_event_0 = time_to_event[0]
    except:
        time_to_event_0 = time_to_event
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'],d_tte['bbox'])):
        if len(seq_obs) < 16 or len(seq_tte) < time_to_event_0:
            remove_index.append(seq_index)
    
    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True):
            del d_obs[k][j]
            del d_tte[k][j]
                
    return d_obs, d_tte
    

def normalize_bbox(dataset, width = 1920, height = 1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))
    
    return normalized_set
            
def prepare_label(dataset):
    labels = np.zeros(len(dataset), dtype='int64') 
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]           
        
    return labels



def batching_np(inp, batch_size):
    batched_list = []
    for i in range(0, len(inp), batch_size):
        if(inp[i: i+batch_size].shape[0] == batch_size):
            batched_list.append(inp[i: i+batch_size])
        
    return np.array(batched_list, dtype='float32')


def pad_sequence(inp_list, max_len):
    padded_sequence = []
    for source in inp_list:
        target = np.zeros((max_len, 4))
        source = source
        target[:source.shape[0], :] = source
        
        padded_sequence.append(target)
        
    return padded_sequence



def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength*progress))
    text = "\r[{}] {:0.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()




def crop_image(image, center, size= (672, 672)):
    
    width = size[0]
    height = size[1]
    
    x = int(center[0])
    y = int(center[1])
    
    h1 = int(height / 2)
    w1 = int(width / 2)

    h2 = int(height / 2)
    w2 = int(width / 2)  
    
    if((x - w1) < 0):
        w1 = x
        w2 = width - x
        
    if((x + w2) > image.shape[1]):
        w2 = image.shape[1] - x
        w1 = width - w2
        
    if((y - h1) < 0):
        h1 = y
        h2 = height - y
        
    if((y + h2) > image.shape[0]):
        h2 = image.shape[0] - y
        h1 = height - h2
    
    crop_img = image[y - h1 : y + h2, x - w1 : x + w2]
    
    
    new_center = [w1, h1]
    
    return crop_img, new_center


def bbox_loc(bbox, new_c):
    
    w = np.abs(int(bbox[2] - bbox[0]))
    h = np.abs(int(bbox[1] - bbox[3]))

    new_bbox = [new_c[0] - w / 2, new_c[1] - h/2, new_c[0] + w/2, new_c[1] + h/2]
    
    return new_bbox
    
    

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
    
    
    
    