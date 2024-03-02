# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import random

def main(args):

    class_label_list = [1908, 4219, 4701, 5345, 5476, 5839]

    valid_save_path = "samples"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    print(args.num_classes)
    print(args.num_super_classes)
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_super_classes=args.num_super_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"

    print(ckpt_path)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # subclass->superclass
    map_list = torch.ones([10000],device=device) * 10 # Animalia
    map_list[5729:9999 + 1] = 0 # Plants
    map_list[175:2700 + 1] = 1 # Insects
    map_list[3111:4596 + 1] = 2 # Birds
    map_list[5388:5728 + 1] = 3 # Fungi
    map_list[4859:5171 + 1] = 4 # Reptiles
    map_list[4613:4858 + 1] = 5 # Mammals
    map_list[2756:2938 + 1] = 6 # Ray-finned Fishes
    map_list[2939:3108 + 1] = 7 # Amphibians
    map_list[5219:5387 + 1] = 8 # Mollusks
    map_list[4:156 + 1] = 9 # Arachnids

    for i in range(len(class_label_list)):
        print(class_label_list[i])
        
        random_number = args.seed
        # random_number = random.randint(0, 10000)
        # print("random_num=" + str(random_number)) 

        # Setup PyTorch:
        torch.manual_seed(random_number)
        torch.set_grad_enabled(False) 
        
        class_labels = [class_label_list[i]]

        map_list.to(device)

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)

        # args.num_classes = 10000  
        y_super = map_list[y].long() + args.num_classes

        # y_super = y
        y = torch.cat([y, y_super], 0)
        
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale) 

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        if not os.path.isdir(valid_save_path ):
            os.makedirs(valid_save_path )

        file_name = "class_"+str(class_label_list[i])+"_sample.png"
        save_image(samples, valid_save_path + os.sep + file_name, nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=10000)
    parser.add_argument("--num-super-classes", type=int, default=11)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="models/FineDiffusion-iNat.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
