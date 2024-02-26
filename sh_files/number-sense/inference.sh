cd runner
python inference.py \
    --model_name_or_path 'your model path' \
    --dataset_name 'NumberSense' \
    --annot_file_path 'your test annotation file path' \
    --decoder_type 'bert-base' \
    --task 'cot' \
    --batch_size 8 \
    --beams 10 \
    --topk 1
    # --use_plugin \
    # --use_type \
    # --use_lora 
    

