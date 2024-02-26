import logging
import os
import sys
import torch
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
from libs.data.transform import CallResizeImage

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



class GeoQACapDataset():
    def __init__(self, annot_file_path, processor):
        self.annot_file_path = annot_file_path
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.annot_list = pickle.load(open(self.annot_file_path, 'rb'))

    def __len__(self):
        return len(self.annot_list)

    def __getitem__(self, idx):      
        data_info = self.annot_list[idx]
        image = data_info["image"] if "image" in list(data_info.keys()) else data_info["img"]
        if len(image.shape) == 2:
            img_arr = np.zeros((3, *image.shape))
            img_arr[...] = np.expand_dims(image, axis=0)
            image = img_arr
        img_transform = CallResizeImage(self.processor)
        info, pixel_values = img_transform(data_info, image)
        decoder_input_ids = self.tokenizer("<caption_task>")['input_ids'][:-1]
        return {
            "data_info": info,
            "decoder_input_ids": [decoder_input_ids],
            "pixel_values": pixel_values
        }



def test_caption():
    model_name_or_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/GeoCaption/swin-bert-base-cased'
    train_annot_path = '/train20/intern/permanent/ycpan4/dataset/geoqa/GeoQA3/test.pk'
    # train_annot_path = '/train20/intern/permanent/ycpan4/dataset/unigeo/UniGeo/proving_test.pk'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    img_processor = DonutImageProcessor.from_pretrained(model_name_or_path)
    processor = DonutProcessor(image_processor=img_processor, tokenizer=tokenizer)

    dataset = GeoQACapDataset(
        annot_file_path=train_annot_path,
        processor=processor
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name_or_path,
        ignore_mismatched_sizes=True
    )
    device = torch.device('cuda')
    model.eval()
    model.to(device)

    for data in dataset:
        data_info = data['data_info']
        input_ids = torch.tensor(data['decoder_input_ids'])
        output_ids = model.generate(
            data['pixel_values'].to(device),
            decoder_input_ids = input_ids.to(device),
            max_length = 512,
            early_stopping = True,
            num_beams = 1,
            # num_return_sequences = self.beams,
            bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()bad_words_ids = [[tokenizer.unk_token_id]],
            bos_token_id = 101,
            eos_token_id = [102],
            return_dict_in_generate = True,
            # output_attentions = self.output_attentions,
        )
        output_ids = output_ids['sequences']
        res_str = [tokenizer.decode(output_ids[i, 1:-1]) for i in range(output_ids.shape[0])]

        print(res_str)
        print(data_info['id'])
        print('-----------------------------')


if __name__ == '__main__':
    test_caption()