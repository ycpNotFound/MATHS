import logging
import os
import sys

import numpy as np
from tqdm import tqdm
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import libs.configs.default as cfg
from libs.data import create_dataset
from libs.data.data_args import DataTrainingArguments
from libs.data.image_processor import DonutImageProcessor
from libs.model.model_args import ModelArguments
from libs.trainers import PreTrainer as Trainer
from libs.model.vision_encoder_decoder.modeling_vision_encoder_decoder import myVisionEncoderDecoderModel

from peft import LoraConfig, TaskType, get_peft_model
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import pickle

def test_unigeo():
    annot_path = '/train20/intern/permanent/ycpan4/dataset/unigeo/UniGeo/calculation_test.pk'
    data = pickle.load(open(annot_path, 'rb'))
    print(len(data))
    print(data[0])
    return 


def test_main():
    tokenizer_path = '/train20/intern/permanent/ycpan4/code/NumberSense/pretrained_model/bert-base-cased-special'
    encoder_model_path = '/train20/intern/permanent/ycpan4/code/NumberSense/pretrained_model/donut-swin-base'
    train_annot_path = '/train20/intern/permanent/ycpan4/dataset/NumberSense-Tiny/annotation/annot_test.json'
    # train_annot_path = '/train20/intern/permanent/ycpan4/dataset/unigeo/calculation_test_processed.json'
    # train_annot_path = '/train20/intern/permanent/ycpan4/dataset/PGDP5K_annoted_2/annotations/pgdp_caption_test_2.json'
    # train_annot_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/unigeo_all_test.json'
    # train_annot_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_qa_test.json'
    # train_annot_path = '/train20/intern/permanent/ycpan4/dataset/unigeo/UniGeo/proving_test.pk'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    img_processor = DonutImageProcessor.from_pretrained(encoder_model_path)
    processor = DonutProcessor(image_processor=img_processor, tokenizer=tokenizer)

    
    train_dataset = create_dataset(
        dataset_name='NumberSense',
        json_path=train_annot_path,
        processor=processor,
        max_length=512,
        task='cot',
        # task='prove', 
        # task='calculate',
        use_plugin=True,
        # use_type=True,
        # language='cn'
    )

    lengths = []
    # data = train_dataset[304]
    for data in tqdm(train_dataset):
        lengths.append(len(data['decoder_input_ids']))
        if 100 in data['decoder_input_ids']:
            print(tokenizer.decode(data['decoder_input_ids']))

    print('max: ', max(lengths))
    print('min: ', min(lengths))
    print('mean: ', np.mean(lengths))


if __name__ == '__main__':
    test_main()
    # test_unigeo()