## FineDiffusion: Scaling up Diffusion Models for Fine-grained Image Generation with 10,000 Classes

[Paper](http://arxiv.org/abs/2212.09748) | [Project Page](https://www.wpeebles.com/DiT) 

![DiT samples](visuals/main_pic.png)

The class-conditional image generation based on diffusion models is renowned for generating high-quality and diverse images. However, most prior efforts focus on generating images for general categories, e.g., 1000 classes in ImageNet-1k. A more challenging task, large-scale fine-grained image generation, remains the boundary to explore. In this work, we present a parameter-efficient strategy, called FineDiffusion, to fine-tune large pre-trained diffusion models scaling to large-scale fine-grained image generation with 10,000 categories. FineDiffusion significantly accelerates training and reduces storage overhead by only finetuning tiered class embedder, bias terms, and normalization layers' parameters. To further improve the image generation quality of fine-grained categories, we propose a novel sampling method for fine-grained image generation, which utilizes superclass-conditioned guidance, specifically tailored for fine-grained categories, to replace the conventional classifier-free guidance sampling. Compared to full fine-tuning, FineDiffusion achieves a remarkable 1.56x training speed-up and requires storing merely 1.77% of the total model parameters, while achieving state-of-the-art FID of 9.776 on image generation of 10,000 classes. Extensive qualitative and quantitative experiments demonstrate the superiority of our method compared to other parameter-efficient fine-tuning methods.

# Getting Started

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/FineDiffusion/FineDiffusion.git
cd FineDiffusion
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```

## Download FineDiffusion models

We fine-tune the DiT-XL/2-256 and DiT-XL/2-512 models on the iNaturalist 2021 mini training set, as well as fine-tune the DiT-XL/2-256 model on the VegFru dataset using FineDiffusion. (Pre-trained DiT model can be download from: [DiT](https://github.com/facebookresearch/DiT))

The trained FineDiffusion models can be downloaded here :

| DiT Model                                                                                                    | Image Resolution | FID    | LPIPS |
| ------------------------------------------------------------------------------------------------------------ | ---------------- | ------ | ----- |
| [FineDiffusion-iNat](https://drive.google.com/file/d/1BdwaIgSrAhI6qSuJkkSOtVrVNh-puNwC/view?usp=sharing)     | 256x256          | 9.776  | 0.721 |
| [FineDiffusion-iNat-512](https://drive.google.com/file/d/1EjMxlp3GbBkiNf9ObtTzT7aiAwc3qoDf/view?usp=sharing) | 512x512          | 9.490  | 0.723 |
| [FineDiffusion-VegFru](https://drive.google.com/file/d/1YZm83UnNbYdM-sw1vBcacc8DUa7r7uD_/view?usp=sharing)                                                                      | 256x256          | 12.382 | 0.667 |

## Sampling FineDiffusion

For example, to sample from our 256x256 FineDiffusion-iNat model, you can use:

```bash
python sample.py --image-size 256 --seed 0 --ckpt models/FineDiffusion-iNat.pt --num-classes 10000 --num-super-classes 11
```

## Training FineDiffusion

We provide a training script for FineDiffusion in [`train.py`](train.py). This script can be used to fine-tune class-conditional DiT models, but it can be easily modified to support other types of conditioning. To launch DiT-XL/2 (256x256) fine-tuning using FineDiffusion with `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-XL/2 --resume pretrained_models/DiT-XL-2-256x256.pt --data-path /path/to/dataset/inat_train_mini --num-classes 10000 --num-super-classes 11
```

## BibTeX

```bibtex
@article{pan2024finediffusion,
  title={FineDiffusion: Scaling up Diffusion Models for Fine-grained Image Generation with 10,000 Classes},
  author={Pan, Ziying and Wang, Kun and Li, Gang and He, Feihong and Li, Xiwang and Lai, Yongxuan},
  journal={arXiv preprint arXiv:2402.18331},
  year={2024}
}
```
