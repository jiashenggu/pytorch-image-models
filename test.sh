export DATA_PATH=/m2/data/imagenet
export OUTPUT_PATH1=output1
export CKPT=/home/gujiashe/pytorch-image-models/models/mobile-former-26m.pth.tar
# torchrun --nproc_per_node=2 train.py $DATA_PATH \
CUDA_VISIBLE_DEVICES=0 python validate.py $DATA_PATH --model mobile_former_26m --checkpoint $CKPT