# Static Key Attention (SKA)

This repository contains the implementation and experimental results for **Static Key Attention (SKA)**, introduced in our **CVPR 2025 Submission #16724**.

## Introduction
We propose **SKA**, a novel modification to the standard Multi-Head Self-Attention (MHSA) mechanism in Vision Transformers (ViTs). By replacing the dynamic query-key interaction with a **trainable static key**, our method:

- Streamlines the computational process of MHSA.
- Maintains or exceeds state-of-the-art (SOTA) performance in specific tasks.
- Proves particularly effective as **intermediate layers** in hierarchical architectures.

SKA highlights the potential of static key mechanisms for efficient and scalable vision models, especially in **small datasets** or hierarchical networks. A variant of SKA, **Convolutional Static Key Attention (CSKA)**, leverages grouped convolutions to further enhance flexibility and efficiency.

---

## Key Features

### 1. Static Key Attention (SKA)
- Replaces dynamically computed keys with a **learned static weight matrix**.
- Reduces computational complexity while retaining dynamic parameterization.
- Fully trainable in an **end-to-end manner**.

### 2. CSKA: A Special Case of SKA
- Incorporates **grouped convolutions** to generate static keys, inspired by multi-headedness in Transformers.
- Efficiently balances performance and computational demands in **hierarchical vision networks**.

### 3. MetaFormer Framework Compatibility
- SKA is integrated into the MetaFormer framework as a **token-mixing module**, enabling a seamless replacement for MHSA.

---

## Implementation

The implementation details for SKA and its variant CSKA can be found in the `ska.py` and `cska.py` files, respectively. Example scripts for training and evaluation are provided.

### Key Components:
- Static key generation with trainable weight matrices.
- Grouped convolution-based static key generation (CSKA).
- Integration within common backbones such as ViT, Swin Transformer, and ConvFormer.

---

## Experiments

We demonstrate the effectiveness of SKA through experiments on **image classification** and **object detection**. Key results include:

### 1. Image Classification
- Evaluated on CIFAR-10, CIFAR-100, and ImageNet datasets.
- Achieved consistent accuracy improvements and computational efficiency.

### 2. Object Detection and Instance Segmentation
- Tested on COCO using Mask R-CNN and Cascade Mask R-CNN frameworks.
- Showed comparable or improved performance over baselines.

### Sample Results:
| Model        | Params (M) | FLOPs (G) | Top-1 Acc. (%) | Dataset    |
|--------------|------------|-----------|----------------|------------|
| SKA-ViT-S    | 9.5        | 1.28      | 84.7 (+1.3)    | CIFAR-10   |
| CSKA-ViT-S   | 10.0       | 1.33      | 87.0 (+3.6)    | CIFAR-10   |
| CSKAFormer   | 28.0       | 8.2       | 83.4 (+0.1)    | ImageNet   |

---

## How to Use

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-repo/ska.git
cd ska
pip install -r requirements.txt
```

## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).


```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Validation

To evaluate our CAFormer-S18 models, run:

```bash
MODEL=caformer_s18
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --checkpoint /path/to/checkpoint 
```

## Train
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.

```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modify code path here

ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model convformer_s18 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0
```
Training (fine-tuning) scripts of other models are shown in [scripts](/scripts/).


## Bibtex
```
@article{yu2024metaformer,
  author={Yu, Weihao and Si, Chenyang and Zhou, Pan and Luo, Mi and Zhou, Yichen and Feng, Jiashi and Yan, Shuicheng and Wang, Xinchao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MetaFormer Baselines for Vision}, 
  year={2024},
  volume={46},
  number={2},
  pages={896-912},
  doi={10.1109/TPAMI.2023.3329173}}
}
```
