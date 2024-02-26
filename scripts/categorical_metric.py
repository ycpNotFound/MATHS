import json
import os
import re
from tqdm import tqdm

annot_path = 'your test annotation file path'

def metric(result_file_path):
    annot_data = json.load(open(annot_path, 'r'))
    result_data = json.load(open(result_file_path, 'r')) 
    print(result_file_path.split('/')[-1], end=', ')
    print(len(result_data))
    if len(result_data) == 10000:
        return None, None, None
    categories = ['combination_holistic', 'combination_analytical', 'composition_holistic', 'composition_analytical', 'partition_holistic', 'partition_analytical']   
    categorical_dict = {
        "combination_holistic": [],
        "combination_analytical": [],
        "composition_holistic": [],
        "composition_analytical": [],
        "partition_holistic": [],
        "partition_analytical": []
    }
    for key, data_info in annot_data.items():
        result = result_data[key]
        prob_type = data_info['prob_type']
        solve_type = data_info['mathematical_conditions'][-2]
        category = f'{prob_type.lower()}_{solve_type}'
        categorical_dict[category].append(result['correct'])

    metrics = []
    all_num, all_correct_num = 0, 0
    for key in categories:
        value = categorical_dict[key]
        metrics.append(sum(value) / len(value))
        all_num += len(value)
        all_correct_num += sum(value)
    mean_acc = int(all_correct_num / all_num * 10000) / 10000
    metrics = [int(x * 10000) / 10000 for x in metrics]
    task = result_file_path.split('/')[-1].split('.')[0]
    return task, mean_acc, metrics

def formal_print(tasks, mean_accs, category_accs):
    max_length = max(len(s) for s in tasks)
    spaces = " " * (max_length - len('Method'))
    print(f'Method{spaces}\tMean\tCombination\tComposition\tPartition')
    print(f'{spaces}\t\t\th\ta\th\ta\th\ta')
    print('-'*100)
    for task, mean_acc, category_acc in zip(tasks, mean_accs, category_accs):
        sub_str = f'{task:<{max_length}}\t{mean_acc}'
        for acc in category_acc:
            sub_str += f'\t{acc}' 
        print(sub_str)
        # print(f'{task}\t{mean_acc}\t{category_acc[0]}\t{category_acc[1]}\t{category_acc[2]}\t{category_acc[3]}\t{category_acc[4]}\t{category_acc[5]}')

def delete_plugin(res_str):
    pattern = r"<plugin>(.*?)</plugin>"
    matches = re.findall(pattern, res_str)
    for match in matches:
        res_str = res_str.replace(f'<plugin>{match}</plugin>', '')
    return res_str
    
def process_metric(result_file_path):
    annot_data = json.load(open(annot_path, 'r'))
    result_data = json.load(open(result_file_path, 'r')) 
    acc = 0
    
    all_num = 0
    correct_expr_num = 0
    correct_num = 0

    expr_flag = 'expr' in result_file_path.split('/')[-1]

    for key, data_info in annot_data.items():
        all_num += 1
        if not result_data[key]['correct']:
            continue
        result = result_data[key]['pred']
        result_exprs = re.split(r",|\:", result)
        
        result_exprs = [item for item in result_exprs if 'panel' not in item]
        result_exprs = [item.split('answer')[0] for item in result_exprs]
        if 'plugin' in result_file_path:
            result_exprs = [delete_plugin(item) for item in result_exprs]
            result_exprs = [item.replace('>>', '') for item in result_exprs]
        result_exprs = [item.replace('x', str(data_info['target'])) if 'x' in item else item for item in result_exprs]
        # replace x -> target
        result_exprs = [''.join(item.split()) for item in result_exprs]
        result_exprs = [item for item in result_exprs if item != '']
        target_exprs = data_info['expression']
        if expr_flag:
            assert len(result_exprs) == 1
            correct_list = [result_exprs[0] in target_exprs]
        else:
            correct_list = [x==y for x,y in zip(result_exprs, target_exprs)]
        if all(correct_list):
            correct_expr_num += 1

        correct_num += 1

    process_acc_in_all = float(correct_expr_num / all_num)
    process_acc_in_correct = float(correct_expr_num / correct_num)
    return process_acc_in_all, process_acc_in_correct


def main(root_path):
    file_names = [f for f in os.listdir(root_path) if '.' in f]

    annot_path_list = [f'{root_path}/{fname}' for fname in  file_names]
    annot_path_list = sorted(annot_path_list)
    tasks, mean_accs, category_accs = [], [], []

    for path in annot_path_list:
        task, mean_acc, category_acc = metric(path)
        
        if task:
            tasks.append(task)
            mean_accs.append(mean_acc)
            category_accs.append(category_acc)

    formal_print(tasks, mean_accs, category_accs)
    
def process_acc_main(file_names, root_path):
    annot_path_list = [f'{root_path}/{fname}' for fname in file_names]
    annot_path_list = sorted(annot_path_list)
    process_accs = []
    for path in annot_path_list:
        acc_in_all, acc_in_correct = process_metric(path)
        acc_in_all = int(acc_in_all * 10000) / 10000
        print(path.split('/')[-1], acc_in_all, acc_in_correct)

def geoqa_metric_main(result_path, annot_path):
    result_data = json.load(open(result_path, 'r'))
    annot_data = json.load(open(annot_path, 'r'))
    num_dict = {
        "angle": 0,
        "length": 0
    }
    correct_dict = {
        "angle": 0,
        "length": 0
    }
    all, correct = 0, 0
    for key, value in result_data.items():
        all += 1
        # info = annot_data[key]['info']
        question = value['label_str'].split('</prompt>')[0].split(',')[-1]
        if 'angle' in question:
            type = 'angle'
        else:
            type = 'length'

        num_dict[type] += 1
        if value['correct']:
            correct += 1
            correct_dict[type] += 1

    print(f"angle: {correct_dict['angle']} {num_dict['angle']} {float(correct_dict['angle'] / num_dict['angle'])}")
    print(f"length: {correct_dict['length']} {num_dict['length']} {float(correct_dict['length'] / num_dict['length'])}")
    print('all: ', float(correct / all))


if __name__ == '__main__':
    # test()
    # main()
    process_acc_main()
    # geoqa_metric_main()