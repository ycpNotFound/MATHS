import multiprocessing
import numpy as np
import os
import json
from tqdm import tqdm
from json_formater import NoIndent, MyEncoder

def chunk(lst, chunk_size):
    return [lst[x:x+chunk_size] for x in range(0, len(lst), chunk_size)]


def multi_process_do_sth(file_names, thread_num, sub_process_fn, save_dict, root_path, task='train'):
    '''输入save_dict, 输出save_dict'''
    file_names_list = chunk(file_names, int(len(file_names)/(thread_num)))
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    num_works = len(file_names_list)
    workers = list()
    
    for work_i in range(num_works):
        worker = multiprocessing.Process(
        	target = sub_process_fn,
            args = (
            	result_queue, file_names_list[work_i], root_path, task
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    for i in tqdm(range(num_works)):
        sub_dict = result_queue.get()
        save_dict.update(sub_dict)
            
    return save_dict

def sub_process(result_queue, file_names, root_path, task='train'):
    final_dict = {}
    for file_name in tqdm(file_names):
        image_name = file_name.split('.')[0] + '.jpg'
        image_path = f'{root_path}/images/{task}_set/{image_name}'
        file_path = f'{root_path}/{task}_set/{file_name}'
        data = np.load(file_path)
        annot_dict = {
            "image_path": image_path,
            "file_path": file_path,
            "target": int(data['target']),
            "int_list_1": data['int_list_1'].tolist(),
            "int_list_2": data['int_list_2'].tolist(),
            "int_list_3": data['int_list_3'].tolist(),
            "bboxes": data['bboxes'].tolist(),
            "expression": data['final_expression'].tolist(),
            "prob_type": str(data['prob_type']),
            "geometrial_conditions": data['geometrial_conditions'].tolist(),
            "mathematical_conditions": data['mathematical_conditions'].tolist()
        }
        annot_dict = gernerate_caption(annot_dict, noindent=False)
        final_dict[file_name.split('.')[0]] = annot_dict
        # except:
        #     print(f'error at {file_name}')
        #     continue
        
    result_queue.put(final_dict)

def multiprocess_generate_annot_files(root_path, task='train'):
    annot_path = f'{root_path}/annotation'
    if not os.path.exists(annot_path):
        os.makedirs(annot_path)
    file_names = os.listdir(f'{root_path}/{task}_set')
    final_dict = {}
    final_dict = multi_process_do_sth(
        file_names, 
        thread_num=40,
        sub_process_fn=sub_process,
        save_dict=final_dict, 
        root_path=root_path,
        task=task
    )
    with open(f'{annot_path}/annot_{task}.json', 'w') as fp:
        # fp.write(json.dumps(final_dict, cls=MyEncoder, indent=4, ensure_ascii=False))
        fp.write(json.dumps(final_dict, indent=4, ensure_ascii=False))



def generate_annot_files(root_path, task='train'):
    # too slow to annot
    annot_path = f'{root_path}/annotation'
    if not os.path.exists(annot_path):
        os.makedirs(annot_path)
    file_names = os.listdir(f'{root_path}/{task}_set')
    cnt = 0
    final_dict = {}
    for file_name in tqdm(file_names):
        image_name = file_name.split('.')[0] + '.jpg'
        image_path = f'{root_path}/images/{task}_set/{image_name}'
        file_path = f'{root_path}/{task}_set/{file_name}'
        data = np.load(file_path)
        '''
        print('=============================================')
        print('target: ', data['target'])
        print('int list 1: ', data['int_list_1'])
        print('int list 2: ', data['int_list_2'])
        print('int list 3: ', data['int_list_3'])
        print('expression: ', data['final_expression'])
        print('prob_type: ', data['prob_type'])
        print('math_operators: ', data['math_operators'])
        print('geometrial_conditions: ', data['geometrial_conditions'])
        print('mathematical_conditions: ', data['mathematical_conditions'] )
        '''
        
        save_dict = {
            "image_path": image_path,
            "file_path": file_path,
            "target": int(data['target']),
            "int_list_1": NoIndent(data['int_list_1'].tolist()),
            "int_list_2": NoIndent(data['int_list_2'].tolist()),
            "int_list_3": NoIndent(data['int_list_3'].tolist()),
            "bboxes": NoIndent(data['bboxes'].tolist()),
            "expression": NoIndent(data['final_expression'].tolist()),
            "prob_type": str(data['prob_type']),
            "math_operators": NoIndent(data['math_operators'].tolist()),
            "geometrial_conditions": NoIndent(data['geometrial_conditions'].tolist()),
            "mathematical_conditions": NoIndent(data['mathematical_conditions'].tolist())
        }
        save_dict = gernerate_caption(save_dict, noindent=True)
        final_dict[str(cnt)] = save_dict
        cnt += 1

    with open(f'{annot_path}/annot_{task}.json', 'w') as fp:
        fp.write(json.dumps(final_dict, cls=MyEncoder, indent=4, ensure_ascii=False))

def gernerate_caption(save_dict, noindent=False):
    '''
    in the first panel: 
        97-2-2-1=<plugin>97-2-2-1>></plugin>92, 
        99-1-2-4=<plugin>99-1-2-4>></plugin>92, 
    in the second panel: 
        98-2-2-2=<plugin>98-2-2-2>></plugin>92, 
        99-1-1-5=<plugin>99-1-1-5>></plugin>92, 
    so in the third panel: 
        99-1-x-2=92<plugin>99-1-x-2=92>></plugin>answer x=1, 
        97-3-1-1<plugin>97-3-1-1>></plugin>=92
    '''
    res_string = ''
    expr_num_per_panel = int(save_dict['mathematical_conditions'].value[-1]) if noindent else int(save_dict['mathematical_conditions'][-1])
    if expr_num_per_panel == 0: expr_num_per_panel = 1
    exprs = save_dict['expression'].value if noindent else save_dict['expression']
    target = save_dict['target']
    exprs_1 = exprs[:expr_num_per_panel]
    exprs_2 = exprs[expr_num_per_panel:2*expr_num_per_panel]
    exprs_3 = exprs[2*expr_num_per_panel:3*expr_num_per_panel]
    
    int_list_3 = save_dict['int_list_3'].value if noindent else save_dict['int_list_3']
    exprs_3 = modify_marked_expression(exprs_3, int_list_3)
    
    res_string += 'in the first panel: '
    for expr in exprs_1:
        (left_expr, right_expr) = expr.split('=')
        str_i = f'{left_expr}=<plugin>{left_expr}>></plugin>{right_expr}, '
        res_string += str_i
        
    res_string += 'in the second panel: '
    for expr in exprs_2:
        (left_expr, right_expr) = expr.split('=')
        str_i = f'{left_expr}=<plugin>{left_expr}>></plugin>{right_expr}, '
        res_string += str_i
    
    res_string += 'so in the third panel: '
    for expr in exprs_3:
        if 'x' in expr:
            str_i = f'{expr}<plugin>{expr}>></plugin>answer x={target}, '
        else:
            (left_expr, right_expr) = expr.split('=')
            str_i = f'{left_expr}=<plugin>{left_expr}>></plugin>{right_expr}, '
        res_string += str_i
        
    save_dict['captioning'] = res_string
    return save_dict
    
    
def modify_marked_expression(exprs, int_list):
    '''
    将mark位置的数字改成x
    '''
    mark_idx = int_list.index('mark')
    nums_per_expr = int(len(int_list)/len(exprs))
    expr_idx = mark_idx // nums_per_expr
    mark_idx_in_expr = mark_idx % nums_per_expr
    
    symbols = split_expr(exprs[expr_idx])
    
    cnt, mark_idx = 0, 0
    for idx, symbol in enumerate(symbols):
        if symbol in '+-*/=()': continue
        if cnt == mark_idx_in_expr:
            mark_idx = idx
        cnt += 1

    symbols[mark_idx] = 'x'
    exprs[expr_idx] = ''.join(symbols)
    return exprs

    
def parse_exprs_to_nums(exprs):
    num_list = []
    if type(exprs) == list:
        for expr in exprs:
            expr = expr.split('=')[0]
            expr = expr.replace('+', ' ').replace('-', ' ').replace('*', ' ')
            expr = expr.replace('/', ' ').replace('(', ' ').replace(')', ' ')
            expr = expr.split(' ')
            expr = [x for x in expr if x != '']
            num_list += expr
    elif type(exprs) == str:
        exprs = exprs.split('=')[0]
        exprs = exprs.replace('+', ' ').replace('-', ' ').replace('*', ' ')
        exprs = exprs.replace('/', ' ').replace('(', ' ').replace(')', ' ')
        exprs = exprs.split(' ')
        exprs = [x for x in exprs if x != '']
        num_list = exprs
    return num_list

def split_expr(expr: str):
    (left_expr, right_expr) = expr.split('=')
    left_expr = left_expr.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ')
    left_expr = left_expr.replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ')
    left_expr = left_expr.split(' ')
    left_expr = [x for x in left_expr if x != '' or x != ' ']
    return left_expr + ['=', right_expr]