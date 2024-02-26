#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import sys
import re
import numpy as np
import json
from tqdm import tqdm
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
import argparse
import libs.configs.default as cfg
from libs.data import create_dataset
from libs.data.image_processor import DonutImageProcessor
from libs.model.vision_encoder_decoder.modeling_vision_encoder_decoder import myVisionEncoderDecoderModel
from libs.utils.parse_results import parse_for_geoqa, parse_for_ns, parse_for_uniprv

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    DonutProcessor,
    set_seed,
)


def chunk(l,chunk_size):
    return [l[x:x + chunk_size] for x in range(0,len(l),chunk_size)]


class NumberSenseDemo():
    def __init__(
            self, 
            model_name_or_path, 
            dataset_name, 
            annot_file_path, 
            task='cot',
            use_lora=False, 
            use_plugin=False,
            use_type=False,
            debug=False, 
            batch_size=16,
            beams=1,
            topk=1,
            seed=42,
            output_attentions=False,
        ):
        set_seed(seed)
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.annot_file_path = annot_file_path
        self.task = task
        self.debug = debug
        self.use_lora = use_lora
        self.use_plugin = use_plugin
        self.use_type = use_type
        self.device = torch.device('cuda')
        self.bts = batch_size
        self.beams = beams
        self.topk = topk
        self.output_attentions = output_attentions

    def __len__(self):
        return len(self.dataset)

    def load(self):
        print('Loading...')
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.img_processor = DonutImageProcessor.from_pretrained(self.model_name_or_path)
        self.processor = DonutProcessor(image_processor=self.img_processor, tokenizer=self.tokenizer)
        self.dataset = create_dataset(
            dataset_name=self.dataset_name,
            json_path=self.annot_file_path, 
            max_length=512, 
            processor=self.processor, 
            task=self.task, 
            use_plugin=self.use_plugin
        )

        self.model = myVisionEncoderDecoderModel.from_pretrained(
            self.model_name_or_path,
            ignore_mismatched_sizes=True
        )
        self.model.eval()
        self.model.to(self.device)
        index_list = list(range(len(self.dataset)))
        self.batch_idx_list = chunk(index_list, self.bts)
        print('Finish load. length of dataset: {}'.format(len(self.dataset)))
        

    def get_batch_data(self, idx_list):
        batch_data =[self.dataset[idx] for idx in idx_list]
        decoder_input_ids = [data['decoder_input_ids'] for data in batch_data]
        prompt_ids = [data['prompt_ids'] for data in batch_data]
        data_info = [data['data_info'] for data in batch_data]
        pixel_values = torch.cat([data['pixel_values'] for data in batch_data], dim=0)

        return dict(
            pixel_values = pixel_values,
            prompt_ids = prompt_ids,
            decoder_input_ids = decoder_input_ids,
            data_info = data_info,
        )
        

    def inference(self, idx_list, topk):
        data = self.get_batch_data(idx_list)
        input_ids = torch.tensor(data['prompt_ids'])
        data_info = data['data_info']
        bos_token_id = 101
        # eos_token_id = 2
        eos_token_id = 102
        output_ids = self.model.generate(
            data['pixel_values'].to(self.device),
            decoder_input_ids = input_ids.to(self.device),
            max_length = 256,
            early_stopping = True,
            num_beams = self.beams,
            num_return_sequences = self.beams,
            bad_words_ids = [[self.tokenizer.unk_token_id]],
            bos_token_id = bos_token_id,
            eos_token_id = [eos_token_id, 1],
            return_dict_in_generate = True,
            output_scores = True,
            output_attentions = self.output_attentions,
        )
        
        if topk != 0:
            indices = torch.tensor([], dtype=int)
            for i in range(self.bts):
                topk_indices = output_ids['sequences_scores'][i*self.beams: (i+1)*self.beams].to('cpu').topk(topk)[1]
                topk_indices += self.beams*i
                indices = torch.cat((indices, topk_indices), dim=0) 

            output_ids = output_ids['sequences'][indices, :]
        else:
            output_ids = output_ids['sequences']

        res_str = [self.tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]
        label_str = ''
        if 'decoder_input_ids' in data and None not in data['decoder_input_ids']:
            label_str = [self.tokenizer.decode(data['decoder_input_ids'][i][1:-1]) for i in range(len(idx_list))]
        
        return res_str, label_str, data_info


def chunk(l,chunk_size):
    return [l[x:x + chunk_size] for x in range(0,len(l),chunk_size)]



def inference_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='NumberSense')
    parser.add_argument('--annot_file_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default='cot')
    parser.add_argument('--batch_size', type=int,  default=8)
    parser.add_argument('--beams', type=int,  default=1)
    parser.add_argument('--topk',  type=int,  default=0)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--use_lora', action="store_true",  default=False)
    parser.add_argument('--use_plugin', action="store_true", default=False)
    parser.add_argument('--use_type', action="store_true",  default=False)
    parser.add_argument('--output_attentions', action="store_true",  default=False)
    

    args = parser.parse_args()
    task_name = args.model_name_or_path.split('/')[-1]
    target_dataset = args.model_name_or_path.split('/')[-2]
    print(f'tas_name: {task_name}')
    demo = NumberSenseDemo(
        model_name_or_path = args.model_name_or_path,
        dataset_name = args.dataset_name, 
        annot_file_path = args.annot_file_path,
        task = args.task,
        debug = args.debug,
        use_lora = args.use_lora,
        use_plugin = args.use_plugin,
        use_type = args.use_type,
        batch_size = args.batch_size,
        beams = args.beams, 
        topk = args.topk,
        seed = args.seed,
        output_attentions = args.output_attentions,
    )
    demo.load()
    all_num, correct_num = 0, 0
    save_dict = {}

    for idxs in tqdm(demo.batch_idx_list):
        res_str, label_str, data_info = demo.inference(idxs, topk=args.topk)
        if args.dataset_name == 'NumberSense':
            res_dict, correct_list = parse_for_ns(res_str, label_str, data_info, topk=args.topk, task=args.task)
        elif args.dataset_name == 'GeoQA':
            res_dict, correct_list = parse_for_geoqa(res_str, label_str, data_info, args.task)
        elif args.dataset_name == 'UniGeoProve':
            res_dict, correct_list = parse_for_uniprv(res_str, label_str, data_info, args.task)
        else:
            continue
        
        all_num += len(correct_list)
        correct_num += sum(correct_list)
        save_dict.update(res_dict)
        if all_num % 100 == 0:  
            print(f'\nall num: {all_num}')
            print(f'correct: {correct_num}')
            print(f'acc: {correct_num / all_num}')
            torch.cuda.empty_cache()
        if demo.debug:
            print(res_str)
            print(label_str)
            print('---------------------------')

    print(f'all num: {all_num}')
    print(f'correct: {correct_num}')
    print(f'acc: {correct_num / all_num}')

    
    save_dir = f'{args.result_dir}/{target_dataset}/results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/{task_name}_{args.dataset_name}_beam{args.beams}_top{args.topk}.json'
    with open(save_path, 'w') as f:
        f.write(json.dumps(save_dict, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    inference_main()
