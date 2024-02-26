

# add your paths of datasets
NS_train_path = ''
NS_valid_path = ''
NS_test_path = ''

GeoQA_train_path = ''
GeoQA_valid_path = ''
GeoQA_test_path = ''

UniGeoAll_train_path = ''
UniGeoAll_valid_path = ''
UniGeoAll_test_path = ''

UniGeoProve_train_path = ''
UniGeoProve_valid_path = ''
UniGeoProve_test_path = ''

# dataset path dict
datasets = {
    "NumberSense": {
        "train_path": NS_train_path,
        "valid_path": NS_valid_path,
        "test_path": NS_test_path
    },
    "GeoQA": {
        "train_path": GeoQA_train_path,
        "valid_path": GeoQA_valid_path,
        "test_path": GeoQA_test_path
    },
    "UniGeo": {
        "train_path": UniGeoAll_train_path,
        "valid_path": UniGeoAll_valid_path,
        "test_path": UniGeoAll_test_path
    },
    "UniGeoProve": {
        "train_path": UniGeoProve_train_path,
        "valid_path": UniGeoProve_valid_path,
        "test_path": UniGeoAll_test_path
    }
}
