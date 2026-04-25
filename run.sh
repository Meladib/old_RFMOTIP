CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 train.py \
    --data-root /data/pos+mot/Datadir/ \
    --exp-name motip_rfdetr_dancetrack \
    --config-path ./configs/rf_motip_DT_V0motip.yaml

