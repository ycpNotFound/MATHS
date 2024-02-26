import re
import json
from tqdm import tqdm
import random


def parse(logic_forms, stru_seqs: list, sem_seqs: list):
    circle_dict = {}
    for form_str in logic_forms:
    
        if 'Equals(LengthOf(Line' in form_str:
            regex = r'Equals\(LengthOf\(Line\(([^,]+), ([^,]+)\)\), (.+?)\)'
            mts = re.findall(regex, form_str)[0]
            sem_seqs.append(f"{mts[0]}{mts[1]} = {mts[2]}")
            
        elif 'Equals(MeasureOf(Angle(' in form_str:
            if 'angle' in form_str:
                regex = r'Equals\(MeasureOf\(Angle\(([^,]+), ([^,]+), ([^,]+)\)\), MeasureOf\(angle (.+?)\)\)'
                mts = re.findall(regex, form_str)[0]
                sem_seqs.append(f"\\angle {mts[0]}{mts[1]}{mts[2]} = \\angle {mts[3]}")
            elif ', Angle(' in form_str:
                regex = r'Equals\(MeasureOf\(Angle\(([^,]+), ([^,]+), ([^,]+)\)\), Angle\((.+?)\)\)'
                mts = re.findall(regex, form_str)[0]
                sem_seqs.append(f"\\angle {mts[0]}{mts[1]}{mts[2]} = \\angle {mts[3]}")
            else:
                regex = r'Equals\(MeasureOf\(Angle\(([^,]+), ([^,]+), ([^,]+)\)\), (.+?)\)'
                mts = re.findall(regex, form_str)[0]
                sem_seqs.append(f"\\angle {mts[0]}{mts[1]}{mts[2]} = {mts[3]}")
                
        elif 'Equals(MeasureOf(Arc' in form_str:
            regex = r'Equals\(MeasureOf\(Arc\(([^,]+), ([^,]+)\)\), (.+?)\)'
            mts = re.findall(regex, form_str)[0]
            sem_seqs.append(f"arc {mts[0]}{mts[1]}={mts[2]}")
            
        elif 'PointLiesOnLine' in form_str:
            regex = r'PointLiesOnLine\((.+?), Line\(([^,]+), ([^,]+)\)\)'
            mts = re.findall(regex, form_str)[0]
            stru_seqs.append(f"line {mts[1]} {mts[0]} {mts[2]}")
            lines_to_del = [f"line {mts[1]} {mts[0]}", f"line {mts[0]} {mts[1]}", 
                            f"line {mts[1]} {mts[2]}", f"line {mts[2]} {mts[1]}",
                            f"line {mts[0]} {mts[2]}", f"line {mts[2]} {mts[0]}"]
            for item in lines_to_del:
                if item in stru_seqs:
                    stru_seqs.remove(item)
                    
        elif 'Perpendicular' in form_str:
            regex = r'Perpendicular\(Line\(([^,]+), ([^,]+)\), Line\(([^,]+), ([^,]+)\)\)'
            mts = re.findall(regex, form_str)[0]
            sem_seqs.append(f"line {mts[0]}{mts[1]} \\prep line {mts[2]}{mts[3]}")
            
        elif 'PointLiesOnCircle' in form_str:
            
            regex = r'PointLiesOnCircle\((.+?), Circle\(([^,]+)\)\)'
            mts = re.findall(regex, form_str)[0]
            if mts[1] in list(circle_dict.keys()):
                circle_dict[mts[1]] = f'{circle_dict[mts[1]]} {mts[0]}'
            else:
                circle_dict[mts[1]] = f'\\odot {mts[1]} lieson {mts[0]}'

        
        elif 'Parallel' in form_str:
            regex = r'Parallel\(Line\(([^,]+), ([^,]+)\), Line\(([^,]+), ([^,]+)\)\)'
            mts = re.findall(regex, form_str)[0]
            sem_seqs.append(f"line {mts[0]}{mts[1]} \\parallel line {mts[2]}{mts[3]}")
            
        else:
            print('error')
            print(form_str)
            return None   
    for key, value in circle_dict.items():
        stru_seqs.append(value)
        
    # print('stru_seqs: ', stru_seqs)    
    # print('sem_seqs: ', sem_seqs)
    return stru_seqs, sem_seqs
        
def transfer_logic_2_natual_language():
    json_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_train_4w_v2.json'
    save_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_train_4w.json'
    data = json.load(open(json_path, 'r'))
    
    save_dict = {}
    for key, value in tqdm(data.items()):
        logic_forms = value['diagram_logic_forms']
        stru_seqs = value['line_instances']
        stru_seqs = [f'line {c[0]} {c[1]}' for c in stru_seqs]
        sem_seqs = []
        stru_seqs, sem_seqs = parse(logic_forms, stru_seqs, sem_seqs)
        value['parsing_stru_seqs'] = stru_seqs
        value['parsing_sem_seqs'] = sem_seqs
        value.pop('diagram_xml')
        value.pop('triangle_instances')
        value.pop('quad_instances')
        value['image_path'] = value['image_path'].replace('/yrfs1/intern', '/train20/intern/permanent')
        save_dict[key] = value

    with open(save_path, 'w') as f:
        f.write(json.dumps(save_dict, indent=4, ensure_ascii=False))
        

def dataset_addition(generate_path, pgps_path, save_path):
    generate_data = json.load(open(generate_path, 'r'))
    pgps_data = json.load(open(pgps_path, 'r'))
    pgps_diagram_data = {}

    for key, value in tqdm(pgps_data.items()):
        new_key = value['diagram'].split('.')[0]
        caption_str = '<caption_task>'
        for item in value['parsing_stru_seqs']:
            caption_str += f'{item}, '
        for item in value['parsing_sem_seqs']:
            caption_str += f'{item}, '
        pgps_diagram_data[new_key] = {
            "image_path": f"/train20/intern/permanent/ycpan4/dataset/PGPS9K/Diagram/{value['diagram']}",
            "parsing_stru_seqs": value['parsing_stru_seqs'],
            "parsing_sem_seqs": value['parsing_sem_seqs'],
            "caption_str": caption_str[:-2]
        }

    # generate_data.update(pgps_diagram_data)
    with open(save_path, 'w') as f:
        # f.write(json.dumps(generate_data, indent=4, ensure_ascii=False))
        f.write(json.dumps(pgps_diagram_data, indent=4, ensure_ascii=False))


def dataset_addition_main():
    dataset_addition(
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_test.json',
        '/train20/intern/permanent/ycpan4/dataset/PGPS9K/PGPS9K/test.json',
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_test.json',
    )
    dataset_addition(
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_train_4w.json',
        '/train20/intern/permanent/ycpan4/dataset/PGPS9K/PGPS9K/train.json',
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_train.json',
    )


def add_caption_str(json_path):
    data = json.load(open(json_path, 'r'))

def add_caption_str_main():
    test_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_and_pgps_test.json'
    train_path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/generate_and_pgps_train.json'
    add_caption_str(test_path)
    add_caption_str(train_path)


def find_two_symbols_key():
    task = 'train'
    annot_path = f'/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_qa_{task}.json'
    save_path = f'/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_qa_{task}_tmp.json'
    data_annot = json.load(open(annot_path, 'r'))
    symbols = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    save_data = {}
    for key, value in data_annot.items():
        expr_steps = value['expr_steps']
        for expr in expr_steps:
            in_expr = [item in expr for item in symbols]
            if sum(in_expr) >=2:
                print(key)


def test():
    logic_forms = [
        "PointLiesOnLine(G, Line(B, C))",
        "PointLiesOnLine(W', Line(F, W))",
        "Equals(MeasureOf(Angle(D, C, G)), MeasureOf(angle 4))",
        "Equals(MeasureOf(Angle(B, D, F)), 28)",
        "Equals(LengthOf(Line(M, N)), 3x-4)",
        "Equals(MeasureOf(Arc(A, E)), 160)",
        "Perpendicular(Line(B, G), Line(D, G))",
        "Parallel(Line(D, C), Line(E, B))",
    ]
    for logic_form in logic_forms:
        print('-------------------------')
        parse(logic_form, [], [])
        

def Unigeo_addition():
    tasks = ['test', 'val', 'train']
    save_dir = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons'
    for task in tasks:
        cal_path = f'/train20/intern/permanent/ycpan4/dataset/unigeo/calculation_{task}_processed.json'
        prv_path = f'/train20/intern/permanent/ycpan4/dataset/unigeo/proving_{task}_processed.json'
        data_1 = json.load(open(cal_path, 'r', encoding='utf-8'))
        data_2 = json.load(open(prv_path, 'r', encoding='utf-8'))

        data_all = {}
        for key, value in data_1.items():
            data_all[f'cal_{key}'] = value
        
        for key, value in data_2.items():
            data_all[f'prv_{key}'] = value

        dict_key_ls = list(data_all.keys())
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = data_all.get(key)

        
        with open(f'{save_dir}/unigeo_all_{task}.json', 'w') as f:
            f.write(json.dumps(new_dic, indent=4, ensure_ascii=False))

def tmp():
    task = 'train'
    annot_path = f'/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_qa_{task}.json'
    save_path = f'/train20/intern/permanent/ycpan4/code/NumberSense/V3/jsons/pgps_qa_{task}_tmp.json'
    data_annot = json.load(open(annot_path, 'r'))
    save_data = {}
    for key, value in data_annot.items():
        solving_process = value['solving_process']
        solving_process_p = value['solving_process_plugin']
        numbers = re.findall(r'\d+', solving_process)
        for number in numbers:
            if len(number)>5:
                print(key)
                new_number = number[:3]
                if int(number[4]) >= 5:
                    new_number = str(int(new_number)+1)
                value['solving_process'] = solving_process.replace(number, new_number)
                value['solving_process_plugin'] = solving_process.replace(number, new_number)
                for i in range(len(value['expr_steps'])):
                    value['expr_steps'][i] = value['expr_steps'][i].replace(number, new_number)

            save_data[key] = value
        # value['image_path'] = data_image[new_key]['image_path']

    with open(save_path, 'w') as f:
        f.write(json.dumps(save_data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    # transfer_logic_2_natual_language()
    # dataset_addition_main()
    # add_caption_str_main()
    # tmp()
    # find_two_symbols_key()
    Unigeo_addition()