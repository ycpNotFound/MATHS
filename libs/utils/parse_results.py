import re
import json


def parse_for_ns(res_str, label_str, data_info, topk=1, task='cot'):
    res_str = [item.replace('[SEP]', '').replace('[PAD]', '').strip() for item in res_str]
    res_item = []

    if task == 'direct':
        res_str = [res_str_item.split(' ')[0] for res_str_item in res_str]
        res_item = res_str

    elif task == 'cot' or task == 'expr':
        pattern = re.compile(r'answer x = (\d+)')
        for item in res_str:
            result = ''
            match = re.search(pattern, item)
            if match:
                result = match.group(1)
            res_item.append(result)

    batch_size = int(len(res_str) / topk)
    res_item = [res_item[i*topk: (i+1)*topk] for i in range(batch_size)]
    label_item = [str(item['target']) for item in data_info]
    keys = [item['key'] for item in data_info]

    indices = []
    correct_list = []
    cnt = 0
    for x, y in zip(res_item, label_item):
        correct =  False
        index = cnt * topk
        if y in x:
            correct = True
            index = index + x.index(y)
        indices.append(index)
        correct_list.append(correct)
        cnt += 1

    res_dict = {}
    if label_str == '' or task == 'direct':
        label_str = label_item
    for i, key in enumerate(keys):
        res_dict[key] = {
            'pred': res_str[indices[i]],
            'label': label_str[i],
            'correct': correct_list[i]
        }

    return res_dict, correct_list

def parse_for_geoqa(res_str, label_str, data_info, task='cot'):
    data_info = data_info[0]
    pattern = r"<ans>(.*?)</ans>"
    pred_items = []
    res_str = [item.replace(' _ ', '_') for item in res_str]
    label_str = [item.replace(' _ ', '_') for item in label_str]

    for i in range(len(res_str)):
        res_str[i] = res_str[i].strip()
        res_str[i] = res_str[i].replace('[PAD]', '').replace('[SEP]', '').strip()
        if task == 'symbolic':
            res_ans_i = res_str[i].split('</prompt>')[-1].strip()
            res_ans_i = res_ans_i.replace('.', '').strip()
            pred_items.append(res_ans_i)
        else:
            matches = re.findall(pattern, res_str[i]) # find all matches in the string
            if matches:
                pred_num = matches[-1] # get the last match
                pred_num = pred_num.replace(' ', '')
                try:
                    pred_num = float(pred_num)
                except:
                    pass
                pred_items.append(pred_num)

    tgt_ans = data_info['info']['target_number']
    if task == 'symbolic':
        tgt_ans = label_str[0].split('</prompt>')[-1].strip()
    res_dict = {}
    pred_str = res_str[0]
    correct = tgt_ans in pred_items
    if correct:
        pred_str = res_str[pred_items.index(tgt_ans)]

    res_dict[data_info['key']] = {
        "pred_str": pred_str,
        "label_str": label_str[0],
        "pred_num": pred_items,
        "correct": correct,
    }

    return res_dict, [correct]



def parse_for_uniprv(res_str, label_str, data_info, task='cot'):
    data_info = data_info[0]
    label_str = label_str[0]
    if task == 'symbolic':
        label_str = label_str.split('</prompt>')[-1].strip()
        label_str = ''.join(label_str.split(' '))
    else:
        # label_str_list = label_str.split('By')[1:]
        # label_str_list = [item.split(',')[1].strip() for item in label_str_list]
        label_str = label_str.split('</prompt>')[-1]
        label_str_items = re.split(r",|¥.", label_str)
        label_str_items = [item for item in label_str_items if item != '']
        label_reasons, label_states = [], []
        for item in label_str_items:
            if 'By' in item:
                label_reasons.append(item.split('By')[-1].strip())
            else:
                label_states.append(item.strip())

    res_str = [res_str_i.replace('ﾂｰ', '').replace('[SEP]', '').replace('[PAD]', '') for res_str_i in res_str]
    if task == 'symbolic':
        res_str = [item.split('</prompt>')[-1].strip() for item in res_str]
        res_str = [''.join(item.split(' ')) for item in res_str]
    correct = False
    for i, res_str_i in enumerate(res_str):
        try:
            if task == 'symbolic':
                if res_str_i == label_str:
                    correct = True
                    break
            else:
                res_str_i = res_str_i.split('</prompt>')[-1]
                res_str_items = re.split(r",|\.", res_str_i)
                res_str_items = [item for item in res_str_items if item != '']
                res_reasons, res_states = [], []
                for item in res_str_items:
                    if 'By' in item:
                        res_reasons.append(item.split('By')[-1].strip())
                    else:
                        res_states.append(item.strip())
                correct_reason = [x==y for x,y in zip(label_reasons, res_reasons)]
                correct_states = [x==y for x,y in zip(label_states, res_states)]
                correct = all(correct_reason + correct_states)
                if correct:
                    break
        except:
            print('error: ', res_str_i)
            continue

    res_dict = {}
    res_dict[data_info['key']] = {
        "pred_str": res_str_i,
        "label_str": label_str,
        "correct": correct,
    }

    return res_dict, [correct]

    