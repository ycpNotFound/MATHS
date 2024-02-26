source ./.bashrc

if [ -f /.dockerenv ]; then
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/intern/ycpan4/libs/usr_lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/intern/ycpan4/miniconda3/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/intern/ycpan4/miniconda3/envs/donut/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cv6/frwang/libs/usr_lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sppro/hangwu4/lib64
fi

echo $PATH
echo $LD_LIBRARY_PATH

nvidia-smi

cd /train20/intern/permanent/ycpan4/code/NumberSense/V3/runner
source /home4/intern/ycpan4/miniconda3/bin/activate mns

python inference.py \
    --model_name_or_path /train20/intern/permanent/ycpan4/code/NumberSense/V3/experiments/NumberSense/swin-bert-expr \
    --dataset_name 'NumberSense' \
    --annot_file_path /train20/intern/permanent/ycpan4/dataset/NumberSense/annotation/annot_test.json \
    --decoder_type 'bert-base' \
    --task 'expr' \
    --batch_size 8 \
    --beams 10 \
    --topk 1
    # --use_plugin \
    # --use_type \
    # --use_lora 
    

