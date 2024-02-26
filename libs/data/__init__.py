
from . import transform as T
from .dataset import (
    NumberSenseDataset, 
    GeoQADataset, 
    UniGeoProveDataset,
    DataCollator
)

from . import bbox_coding

def create_dataset(
    dataset_name,
    json_path, 
    processor, 
    max_length=512,
    task='cot', 
    use_plugin=False
):

    assert dataset_name in ['NumberSense', 'GeoQA', 'UniGeo', 'UniGeoProve']
    assert task in ['direct', 'expr', 'cot', 'calculate', 'prove']

    tokenize_method_dict = {
        "NumberSense": {
            "direct": T.CallTokenizedInputNSDirect,
            "expr": T.CallTokenizedInputNSExpr,
            "cot": T.CallTokenizedInputNSCoT,
        },
        "GeoQA": {
            "cot": T.CallTokenizedInputGeoQA,
        },
        "UniGeo": {
            "cot": T.CallTokenizedInputUniGeo,
        },
        "UniGeoProve": {
            "cot": T.CallTokenizedInputUniGeoPrv,
        },
    }
    dataset_class_dict = {
        "NumberSense": NumberSenseDataset,
        "GeoQA": GeoQADataset,
        "UniGeo": GeoQADataset,
        "UniGeoProve": UniGeoProveDataset,
    }
    tokenize_method = tokenize_method_dict[dataset_name][task]
    dataset_class = dataset_class_dict[dataset_name]
    transforms = T.Compose([
        T.CallResizeImage(processor), 
        tokenize_method(
            processor, 
            max_length=max_length,
            use_plugin=use_plugin
        ),
    ])

    dataset = dataset_class(json_path, transforms, tokenizer=processor.tokenizer)

    return dataset

    