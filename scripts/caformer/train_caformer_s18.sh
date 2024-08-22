DATA_PATH=~/dev/metaformer/imagenet
CODE_PATH=~/dev/metaformer # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=1
GRAD_ACCUM_STEPS=128 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && ls && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model caformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0