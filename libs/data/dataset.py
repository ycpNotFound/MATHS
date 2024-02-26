# -*- coding: utf-8 -*-


import glob
import os
import copy
import json
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset

from PIL import Image

class NumberSenseDataset(Dataset):
    def __init__(self, annot_file_path, transforms, tokenizer):
        self.annot_file_path = annot_file_path
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.annot_dict = json.load(open(self.annot_file_path, 'r'))
        self.file_names = list(self.annot_dict.keys())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]        
        data_info = self.annot_dict[file_name]
        data_info['key'] = file_name
        data_path = data_info['file_path']
        data = np.load(data_path)
        image = data["image"]
        input_ids, pixel_values = self.transforms(data_info, image)

        # [self.tokenizer.decode(x) for x in labels if x > 0]
        # [self.tokenizer.decode(x) for x in decoder_input_ids]
        labels = input_ids['labels'] if 'labels' in input_ids else None
        decoder_input_ids = input_ids['decoder_input_ids'] if 'decoder_input_ids' in input_ids else None
        prompt_ids = input_ids['prompt_ids'] if 'prompt_ids' in input_ids else None

        return dict(
            labels = labels,
            pixel_values = pixel_values,
            decoder_input_ids = decoder_input_ids,
            prompt_ids = prompt_ids, 
            data_info = data_info,
        )
        # except Exception as e:
        #     print('Error occured while load data: %d' % idx)
        #     print(self.annot_dict[str(idx)])
        #     raise e


class GeoQADataset(Dataset):
    def __init__(self, annot_file_path, transforms, tokenizer, annot_input_path=None):
        self.annot_file_path = annot_file_path
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.annot_dict = json.load(open(self.annot_file_path, 'r', encoding='utf-8'))
        self.index_dict = {}
        for index, key in enumerate(self.annot_dict):
            self.index_dict[index] = key

    def __len__(self):
        return len(self.annot_dict)

    def __getitem__(self, idx):      
        key = self.index_dict[idx]
        data_info = copy.deepcopy(self.annot_dict[key])
        data_info['key'] = key
        image = np.array(Image.open(data_info['image_path']).convert("RGB"))    
        input_ids, pixel_values = self.transforms(data_info, image)

        # [self.tokenizer.decode(x) for x in labels if x > 0]
        # [self.tokenizer.decode(x) for x in decoder_input_ids]
        labels = input_ids['labels']
        decoder_input_ids = input_ids['decoder_input_ids']
        prompt_ids = input_ids['prompt_ids'] if 'prompt_ids' in input_ids else None

        return dict(
            labels = labels,
            pixel_values = pixel_values,
            decoder_input_ids = decoder_input_ids,
            prompt_ids = prompt_ids, 
            data_info = data_info,
        )



class UniGeoProveDataset():
    def __init__(self, annot_file_path, transforms, tokenizer):
        self.annot_file_path = annot_file_path
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.annot_dict = json.load(open(self.annot_file_path, 'r', encoding='utf-8'))
        self.index_dict = {}
        for index, key in enumerate(self.annot_dict):
            self.index_dict[index] = key

    def __len__(self):
        return len(self.annot_dict)

    def __getitem__(self, idx):      
        key = self.index_dict[idx]
        data_info = copy.deepcopy(self.annot_dict[key])
        data_info['key'] = key
        image = np.array(Image.open(data_info['image_path']).convert("RGB"))    
        input_ids, pixel_values = self.transforms(data_info, image)

        # [self.tokenizer.decode(x) for x in labels if x > 0]
        # [self.tokenizer.decode(x) for x in decoder_input_ids]
        labels = input_ids['labels']
        decoder_input_ids = input_ids['decoder_input_ids']
        prompt_ids = input_ids['prompt_ids'] if 'prompt_ids' in input_ids else None

        return dict(
            labels = labels,
            pixel_values = pixel_values,
            decoder_input_ids = decoder_input_ids,
            prompt_ids = prompt_ids, 
            data_info = data_info,
        )


        

class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_data):

        def merge1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = s
            return out
    
        def merge2d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1]] = s
            return out

        def merge3d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            dim3 = max([s.shape[2] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
            return out

        def mask1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = 1
            return out

        pixel_values = torch.cat([data['pixel_values'] for data in batch_data], dim=0)
        label_ids = merge1d([torch.tensor(data['labels']) for data in batch_data], -100)
        # bert pad token id -> 0
        decoder_input_ids = merge1d([torch.tensor(data['decoder_input_ids']) for data in batch_data], 0)

        decoder_attention_mask = torch.tensor([[1 if idx != 0 else 0 for idx in data ] for data in decoder_input_ids])
        # decoder_token_type_ids = merge1d([torch.tensor(data['decoder_token_type_ids']) for data in batch_data], 0)
        return {
            "pixel_values": pixel_values,
            "labels": label_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            # "decoder_token_type_ids": decoder_token_type_ids, 
            "return_dict": False
        }