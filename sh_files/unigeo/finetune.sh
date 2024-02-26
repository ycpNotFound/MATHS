cd runner
LOCAL_RANK=0,1
CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export NNODES=2

export dataset_name='UniGeo'
export decoder_task='cot'
export use_plugin=False
export use_lora=False
export use_type=False   

export model_name_or_path='your model path'
export encoder_path=donut-swin-base
export decoder_path=bert-base-cased
export output_dir='your model path'

export gradient_accumulation_steps=2
export per_device_train_batch_size=4
export per_device_eval_batch_size=4
export dataloader_num_workers=4
export num_train_epochs=50
export learning_rate=5e-5
export master_port=10009

if [[ $NNODES -gt 1 ]]; then
    python -m torch.distributed.launch --use-env --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        main.py \
        --seed 3407 \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --encoder_path $encoder_path \
        --decoder_path $decoder_path \
        --output_dir $output_dir \
        --decoder_task $decoder_task \
        --use_plugin $use_plugin \
        --use_type $use_type \
        --use_lora $use_lora \
        --do_train \
        --save_strategy steps \
        --save_steps 5000 \
        --logging_strategy steps \
        --logging_steps 100 \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
	    --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio 0.1 \
        --learning_rate $learning_rate \
        --ignore_data_skip True \
        --fp16 \
        # --finetune \
        # --overwrite_output_dir \
        # --do_eval \
        # --evaluation_strategy steps \
        # --eval_steps 2000 \
        
else
	python -m torch.distributed.launch --use-env --nproc_per_node=$NGPUS --master_port=$master_port \
        main.py \
        --seed 3407 \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --encoder_path $encoder_path \
        --decoder_path $decoder_path \
        --output_dir $output_dir \
        --decoder_task $decoder_task \
        --use_plugin $use_plugin \
        --use_type $use_type \
        --use_lora $use_lora \
        --do_train \
        --save_strategy steps \
        --save_steps 5000 \
        --logging_strategy steps \
        --logging_steps 100 \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
	    --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio 0.1 \
        --learning_rate $learning_rate \
        --ignore_data_skip True \
        --fp16 \
        # --finetune \
        # --overwrite_output_dir \
        # --do_eval \
        # --evaluation_strategy steps \
        # --eval_steps 2000 \
        
fi
