import os
# import cv2
import numpy as np
from PIL import Image
import copy
import re

def delete_plugin(res_str):
    pattern = r"<plugin>(.*?)</plugin>"
    matches = re.findall(pattern, res_str)
    for match in matches:
        res_str = res_str.replace(f'<plugin>{match}</plugin>', '')
    
    res_str = res_str.replace('answer', ', answer')
    return res_str

def replace_special_symbols(string):
    if '故选' in string:
        string = string.split('故选')[0]
    replace_items = [
        ('试题分析:', ''),('△', '\\triangle '),
        ('⊙', '\\odot '), ('∠', 'm\\angle '),
        ('⊥', '\\prep '), ('⟂', '\\prep'),
        ('∥', '\\parallel '),
        ('∵', '因为'),('∴', '所以'),
        ('≌', '\\cong '), ('⌒', '\\arc '),
        ('▱', 'parallelogram '), ('α', 'a'), ('β', 'b'),
        ('㎝', 'cm'), ('…', ','), ("\\frac{1}{2}", "1/2"),
        ('~', '\\sim '), ('√', '\\sqrt'), ('≈', '\\approx'),
        ('π', '\\pi '), ('•', '*'), ('∽', '\\sim '), 
        ('□', 'parallelogram ')
    ]
    for pair in replace_items:
        x, y = pair[0], pair[1]
        string = string.replace(x, y)
    return string

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data


class CallTokenizedInputNSDirect():
    # Tokenize for Machine Number Sense, only predict one number
    def __init__(self, processor, max_length=64, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length=max_length

    def __call__(self, info, image):

        answer = str(info['target'])
        decoder_input_ids = self.tokenizer(answer)['input_ids']
        label_ids = decoder_input_ids[1:] + [-100]

        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': decoder_input_ids[:1],
        }
        
        return input_ids, image


class CallTokenizedInputNSExpr():
    # Tokenize for Machine Number Sense, predict expression in last panel
    def __init__(self, processor, max_length=64, use_plugin=False, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.use_plugin = use_plugin

    def __call__(self, info, image):

        answer = str(info['target'])
        caption_str = info['captioning']
        answer_strs = caption_str.split('in the third panel:')[1].split(',')
        answer_str = [item for item in answer_strs if 'answer' in item][0]
        if self.use_plugin:
            answer_str = answer_str.replace('>></plugin>', '</plugin>>>')
        else:
            answer_str = delete_plugin(answer_str).strip()
        decoder_input_ids = self.tokenizer(answer_str)['input_ids']
        label_ids = decoder_input_ids[1:] + [-100]

        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': decoder_input_ids[:1],
        }
        
        return input_ids, image


class CallTokenizedInputNSCoT():
    # Tokenize for Machine Number Sense, predict expressions in three panels
    def __init__(self, processor, max_length=64, use_plugin=False, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length=max_length
        self.use_plugin = use_plugin

    def __call__(self, info, image):
        captioning_str = info['captioning']
        if captioning_str[-2:] == ', ':
            captioning_str = captioning_str[:-2]
            
        if not self.use_plugin:
            captioning_str = delete_plugin(captioning_str)

        captioning_str = captioning_str.replace('>></plugin>', '</plugin>>>')
        captioning_str = captioning_str.replace(f"answer x={info['target']}, ", f"answer x={info['target']}")
        captioning_str = captioning_str.replace(f"answer x={info['target']}", f"answer x={info['target']}, ")
        if captioning_str[-2:] == ', ':
            captioning_str = captioning_str[:-2]
        decoder_input_ids = self.tokenizer(captioning_str)['input_ids']
        decoder_input_ids = decoder_input_ids[:self.max_length]
    
        # debug: [self.tokenizer.decode(x) for x in [i for i in label_ids[1] if i>0]]
        label_ids = decoder_input_ids[1:] + [-100]
        label_ids = label_ids[:self.max_length]

        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': decoder_input_ids[:5],
        }
        
        return input_ids, image


class CallTokenizedInputGeoQA():
    # Tokenize for GeoQA
    def __init__(self, processor, max_length=512, use_plugin=False, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.use_plugin = use_plugin

    def __call__(self, info, image):
        prompt_str = replace_special_symbols(info['problem'])
        # question = question.split('(Elements:')[0]
        prompt_str = f"<calculation_task><prompt>{prompt_str}</prompt>"

        label_str = prompt_str
    
        label_str += info['answer']

        if self.use_plugin:
            label_str += '. exprs: '
            for expr in info['info']['exprs']:
                expr_items = expr.split('=')
                if len(expr_items) != 3:
                    continue
                expr_items[1] = f'<plugin>{expr_items[1]}</plugin>'
                expr = f'{expr_items[0]}={expr_items[1]}={expr_items[2]}'
                label_str += expr
                label_str += ', '

        label_str = replace_special_symbols(label_str)
        label_str = label_str.strip() 
        # if label_str[-1] == ',':
        #     label_str = label_str[:-1]
        label_str = f"{label_str} <ans>{info['info']['target_number']}</ans>"

        prompt_ids = self.tokenizer(prompt_str)['input_ids'][:-1]  
        decoder_input_ids = self.tokenizer(label_str)['input_ids']
        label_ids = [-100] * len(prompt_ids) + decoder_input_ids[len(prompt_ids): ]
        label_ids = label_ids[1:] + [-100]

        label_ids = label_ids[:self.max_length]
        decoder_input_ids = decoder_input_ids[:self.max_length]

        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': prompt_ids,
        }
        return input_ids, image

class CallTokenizedInputUniGeo():
    # Tokenize for UniGeo
    def __init__(self, processor, max_length=512, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, info, image):
        question = replace_special_symbols(info['problem'])
        
        prompt_str = ''
        if 'prv' in info['key']:
            question = question.split('(Elements:')[0]
            prompt_str = f"<prove_task><prompt>{question}</prompt>"
        elif 'cal' in info['key']:
            prompt_str = f"<calculation_task><prompt>{question}</prompt>"

        label_str = prompt_str
        label_str += info['answer']
        label_str = replace_special_symbols(label_str)
        label_str = label_str.strip() 
        if label_str[-1] == ',':
            label_str = label_str[:-1]
        if 'cal' in info['key']:
            label_str = f"{label_str} <ans>{info['info']['target_number']}</ans>"
        
        prompt_ids = self.tokenizer(prompt_str)['input_ids'][:-1]  
        decoder_input_ids = self.tokenizer(label_str)['input_ids']


        label_ids = [-100] * len(prompt_ids) + decoder_input_ids[len(prompt_ids): ]
        label_ids = label_ids[1:] + [-100]

        label_ids = label_ids[:self.max_length]
        decoder_input_ids = decoder_input_ids[:self.max_length]


        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': prompt_ids,
        }
        return input_ids, image


class CallTokenizedInputUniGeoPrv():
    # Tokenize for proving questions of UniGeo
    def __init__(self, processor, max_length=512, **kwargs):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, info, image):
        question = replace_special_symbols(info['problem'])
        question = question.split('(Elements:')[0]
        prompt_str = f"<prove_task><prompt>{question}</prompt>"

        label_str = prompt_str
        label_str += info['answer']
        label_str = replace_special_symbols(label_str)

        prompt_ids = self.tokenizer(prompt_str)['input_ids'][:-1]  
        decoder_input_ids = self.tokenizer(label_str)['input_ids']


        label_ids = [-100] * len(prompt_ids) + decoder_input_ids[len(prompt_ids): ]
        label_ids = label_ids[1:] + [-100]

        label_ids = label_ids[:self.max_length]
        decoder_input_ids = decoder_input_ids[:self.max_length]


        input_ids = {
            'labels': label_ids,
            'decoder_input_ids': decoder_input_ids,
            'prompt_ids': prompt_ids,
        }
        return input_ids, image


class CallResizeImage:
    '''
        Resize input image to a small size to save GPU Memory
    '''
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, info, image):
        
        # image_all = np.concatenate((image[0], image[1], image[2]), axis=1)
        if image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        outputs = self.processor(
            Image.fromarray(np.uint8(image)), 
            return_tensors="pt", 
        )
        pixel_values = outputs['pixel_values']
        return info, pixel_values


