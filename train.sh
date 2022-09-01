export DATA_PATH=/m2/data/imagenet
export OUTPUT_PATH1=output1
export EXP=mobile_former_26m_reimpl
# torchrun --nproc_per_node=2 train.py $DATA_PATH \
CUDA_VISIBLE_DEVICES=0 python train.py $DATA_PATH \
    --output $OUTPUT_PATH1 \
    --model mobile_former_26m \
    -j 8 \
    --batch-size 128 \
    --epochs 450 \
    --opt adamw \
    --sched cosine \
    --lr 0.0008 \
    --weight-decay 0.08 \
    --drop 0.1 \
    --drop-path 0.0 \
    --mixup 0.2 \
    --aa rand-m9-mstd0.5 \
    --remode pixel \
    --reprob 0.0 \
    --color-jitter 0. \
    --log-interval 200 \
    > $EXP.log