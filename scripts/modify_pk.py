import json
import pickle
from tqdm import tqdm

def modify_geoqa():
    task = 'test'
    pk_dir = '/train20/intern/permanent/ycpan4/dataset/geoqa/GeoQA3'
    pk_path = f'{pk_dir}/{task}.pk'
    json_path = f'{pk_dir}/calculation_{task}_en.json'
    pk_data = pickle.load(open(pk_path, 'rb'))
    json_data = json.load(open(json_path, 'r', encoding='utf-8'))
    save_data = []
    for i, pk_item in enumerate(tqdm(pk_data)):
        json_item = json_data[f'{i}']
        # print(pk_item)
        # print(json_item)
        pk_item['exprs'] = json_item['exprs']
        answer_str = json_item['answer_en'].strip()
        if '.' != answer_str[-1]:
            answer_str +='. '
        pk_item['answer_en'] = answer_str
        solving_process = answer_str
        solving_process_plugin = answer_str
        for expr in json_item['exprs']:
            solving_process += expr
            idx = expr.rfind("=")  # 找到最后一个"="的索引
            part1 = expr[:idx]  # 切片得到前半部分
            part2 = expr[idx + 1:]  # 切片得到后半部分

            solving_process_plugin += f'<plugin>{part1}</plugin>={part2}'
            if expr != json_item['exprs'][-1]:
                solving_process += ', '
                solving_process_plugin += ', '

        ans = json_item['target_number']
        solving_process += f'<ans>{str(ans)}</ans>'
        solving_process_plugin += f'<ans>{str(ans)}</ans>'
        pk_item['English_problem'] = json_item['English_problem']
        pk_item['solving_process'] = solving_process
        pk_item['solving_process_plugin'] = solving_process_plugin
        save_data.append(pk_item)


    with open(f'{pk_dir}/{task}_en.pk', 'wb') as f:
        f.write(pickle.dumps(pk_data))

if __name__ == '__main__':
    modify_geoqa()
