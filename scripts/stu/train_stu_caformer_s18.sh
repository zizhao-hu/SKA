DATA_PATH=/project/rostamim_991/zizhao/metaformer/imagenet
CODE_PATH=/project/rostamim_991/zizhao/STU/ # modify code path here
# CKPT_PATH=/project/rostamim_991/zizhao/STU/output/train/20240930-025807-stu_caformer_s18-224/checkpoint-243.pth.tar

ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

cd $CODE_PATH && ls && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model stu_caformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 