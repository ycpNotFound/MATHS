import json
from tqdm import tqdm
def read_results_ns():
    paths = [
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/NumberSense/results/swin-bert-cot-plugin_NumberSense_beam10_top1.json',
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/NumberSense/results/swin-bert-cot_NumberSense_beam10_top1_all.json',
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/NumberSense/results/swin-bert-expr_NumberSense_beam10_top1.json',
        '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/NumberSense/results/swin-bert-expr-plugin_NumberSense_beam10_top1.json'
    ]
    datas = [
        json.load(open(path_i, 'r')) for path_i in paths
    ]


    for key in tqdm(datas[0]):
        flags = []
        results = [data[key] for data in datas]
        expr_1 = results[0]['pred'].split('=')[0].replace(' ', '')
        expr_2 = results[1]['pred'].split('=')[0].replace(' ', '')
        flags.append(expr_1 == expr_2)
        flags.append('holistic' in key)
        flags.append('combination' in key)
        flags.append(results[0]['correct'] == True)
        flags.append(results[1]['correct'] == False)
        flags.append(results[2]['correct'] == False)
        flags.append(results[3]['correct'] == False)

        label = results[2]['label'].split(',')[0]
        
        # flags.append()

        if all(flags):
            print(key)
            print('plugin: ', results[0]['pred'])
            print('direct: ', results[1]['pred'])
            print('expr:   ', results[2]['pred'])
            print('expr-p: ', results[3]['pred'])
            print('label: ', results[2]['label'])
            print('--------------------------------')

def read_results_geoqa():
    path = '/train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/GeoQA/results/cased-cot-direct-3_GeoQA_beam10_top10.json'
    data = json.load(open(path, 'r'))
    for key, value in data.items():
        flags = []
        flags.append(len(value['pred_str']) < 300)
        flags.append(value['correct'] == False)
        if all(flags):
            print(key)
            print('pred: ', value['pred_str'])
            print('label: ', value['label_str'])
            print('------------------------------------')


if __name__ == '__main__':
    read_results_geoqa()
    