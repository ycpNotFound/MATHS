python inference.py \
    --model_name_or_path 'your model path' \
    --dataset_name 'UniGeoProve' \
    --annot_file_path 'your test annotation file path' \
    --decoder_type 'bert-base' \
    --task 'cot' \
    --batch_size 1 \
    --beams 10 \
    --topk 10
    # --use_plugin \

