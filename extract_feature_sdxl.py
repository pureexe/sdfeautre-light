import argparse
import torch
import os
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from torchvision.transforms import PILToTensor
from diffusers import AutoencoderKL
from relighting.dift_sdxl import SDXLFeaturizer, MyUNet2DConditionModel, OneStepSDXLPipeline
from relighting.dataset import GeneralLoader
from relighting.utils import name2hash

def create_argparser():
    parser = argparse.ArgumentParser(
    description='''extract dift from input image, and save it as torch tenosr,
        in the shape of [c, h, w].''')
    
    parser.add_argument("--img_height", type=int, default=1024, help="Dataset Image Height")
    parser.add_argument("--img_width", type=int, default=1024, help="Dataset Image Width")
    parser.add_argument('--t', default=101, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=None, type=int,
                        help='which upsampling block of U-Net to extract the feature map, None means all')
    parser.add_argument('--prompt', default='a perfect mirrored reflective chrome ball sphere', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument("--negative_prompt", type=str, default="matte, diffuse, flat, dull", help="negative prompt used in the stable diffusion")
    parser.add_argument('--ensemble_size', default=1, type=int, 
                        help='number of repeated images in each batch used to get features')
    parser.add_argument('--input_dir', type=str,
                        help='path to the input image file')
    parser.add_argument('--output_dir', type=str, default='dift.pt',
                        help='path to save the output features as torch tensor')
    parser.add_argument('--global_seed', type=int, default=-1, #TODO: support random seed from filename 
                        help='global seed when adding noise to img_tensor (seed < 0  is random seed by filename)')
    
    return parser

def preprocess(img):
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    img_tensor = img_tensor.to(dtype=torch.float16)
    return img_tensor

def main():
    # load arguments
    args = create_argparser().parse_args()
    
    # create pipeline
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "cuda:0"

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    unet = MyUNet2DConditionModel.from_pretrained(base_id, subfolder="unet", torch_dtype=torch.float16)
    pipe = OneStepSDXLPipeline.from_pretrained(base_id, vae=vae, unet=unet, safety_checker=None)
    dift = SDXLFeaturizer(pipe, device=device)

    # load images
    dataset = GeneralLoader(
        root=args.input_dir, 
        resolution=(args.img_height, args.img_width),
        return_image_path=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for image, image_path in tqdm(dataset):
        img_tensor = preprocess(image)
        
        # check if cnt==3: continue
        cnt = 0
        basename = Path(image_path).stem
        for i in range(3):
            output_path = os.path.join(args.output_dir, f"{basename}_layer{i}.pt")
            if os.pathj.exists(output_path):
                cnt += 1
        if cnt == 3:
            continue
        

        # extract features
        seed = args.global_seed
        if seed < 0:
            filename = os.path.basename(image_path).split(".")[0]
            seed = name2hash(filename) 
        generator = torch.Generator().manual_seed(seed)
        ft = dift.forward(
                img_tensor,
                prompt=args.prompt,
                t=args.t,
                up_ft_index=args.up_ft_index,
                ensemble_size=args.ensemble_size,
                generator=generator
            )
        
        # save features
        basename = Path(image_path).stem
        if isinstance(ft, torch.Tensor):
            print(ft.shape)
            # extract feature from only 1 layer
            output_path = os.path.join(args.output_dir, f"{basename}.pt")
            torch.save(ft.squeeze(0).cpu(), args.output_path) # save feature in the shape of [c, h, w]
        else:
            # extract feature from all layers
            for i, layer_ft in enumerate(ft):
                print(i, layer_ft.shape)
                output_path = os.path.join(args.output_dir, f"{basename}_layer{i}.pt")
                torch.save(layer_ft.squeeze(0).cpu(), output_path)


if __name__ == '__main__':
    main()